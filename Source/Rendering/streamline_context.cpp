#ifdef _WIN32

#include "streamline_context.h"
#include <spdlog/spdlog.h>
#include <windows.h>

#ifdef METALLIC_HAS_STREAMLINE

#include <vulkan/vulkan.h>
#include <sl.h>
#include <sl_dlss.h>
#include <sl_helpers_vk.h>
#include <sl_consts.h>
#include <string_view>

// Streamline SDK version used at link time
static constexpr uint64_t kSlSdkVersion = sl::kSDKVersion;
static constexpr uint32_t kDefaultViewportId = 0;

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

static sl::float4x4 toSlMatrix(const float* colMajor) {
    // Convert column-major (MathLib) to row-major (Streamline)
    sl::float4x4 out{};
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out.row[r].x = 0; // zero-init
    // colMajor[col*4 + row] -> out.row[row][col]
    for (int r = 0; r < 4; r++) {
        float* dst = &out.row[r].x;
        for (int c = 0; c < 4; c++) {
            dst[c] = colMajor[c * 4 + r];
        }
    }
    return out;
}

static sl::Resource makeSlTextureResource(void* image,
                                          void* imageView,
                                          const StreamlineDlssFrameData::VulkanTextureInfo& info) {
    sl::Resource resource(sl::ResourceType::eTex2d,
                          image,
                          nullptr,
                          imageView,
                          info.state);
    resource.width = info.width;
    resource.height = info.height;
    resource.nativeFormat = info.nativeFormat;
    resource.mipLevels = info.mipLevels > 0 ? info.mipLevels : 1;
    resource.arrayLayers = info.arrayLayers > 0 ? info.arrayLayers : 1;
    resource.flags = info.flags;
    resource.usage = info.usage;
    return resource;
}

static HMODULE findStreamlineInterposerModule() {
    HMODULE module = GetModuleHandleW(L"sl.interposer.dll");
    if (!module) {
        module = LoadLibraryW(L"sl.interposer.dll");
    }
    return module;
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
               | sl::PreferenceFlags::eUseManualHooking
               | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
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
    if (HMODULE interposerModule = findStreamlineInterposerModule()) {
        m_vkGetDeviceProcAddrProxy =
            reinterpret_cast<void*>(GetProcAddress(interposerModule, "vkGetDeviceProcAddr"));
    }
    if (!m_vkGetDeviceProcAddrProxy) {
        spdlog::warn("Streamline Vulkan proxy lookup is unavailable; manual present hooks will fall back to native Vulkan");
    }
    spdlog::info("Streamline SDK initialized");
    return true;
}

bool StreamlineContext::queryVulkanRequirements(StreamlineVulkanRequirements& out) const {
    if (!m_initialized) return false;

    sl::FeatureRequirements requirements{};
    sl::Result result = slGetFeatureRequirements(sl::kFeatureDLSS, requirements);
    if (result != sl::Result::eOk) {
        spdlog::warn("slGetFeatureRequirements(DLSS) failed (result={})", static_cast<int>(result));
        return false;
    }

    out.instanceExtensions.clear();
    out.deviceExtensions.clear();
    out.needsTimelineSemaphore = false;

    // Copy required Vulkan instance extensions
    for (uint32_t i = 0; i < requirements.vkNumInstanceExtensions; i++) {
        out.instanceExtensions.push_back(requirements.vkInstanceExtensions[i]);
    }

    // Copy required Vulkan device extensions
    for (uint32_t i = 0; i < requirements.vkNumDeviceExtensions; i++) {
        out.deviceExtensions.push_back(requirements.vkDeviceExtensions[i]);
    }

    // Always request timeline semaphore — DLSS needs it and it's core in Vulkan 1.2+
    out.needsTimelineSemaphore = true;

    // Also ensure VK_KHR_push_descriptor is present (DLSS requirement)
    bool hasPushDescriptor = false;
    for (auto ext : out.deviceExtensions) {
        if (std::string_view(ext) == "VK_KHR_push_descriptor") {
            hasPushDescriptor = true;
            break;
        }
    }
    if (!hasPushDescriptor) {
        out.deviceExtensions.push_back("VK_KHR_push_descriptor");
    }

    spdlog::info("Streamline DLSS requires {} instance extensions, {} device extensions",
                  out.instanceExtensions.size(), out.deviceExtensions.size());
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

    sl::ViewportHandle viewport{kDefaultViewportId};
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

    sl::ViewportHandle viewport{kDefaultViewportId};

    // Set per-frame constants
    sl::Constants constants{};
    constants.cameraViewToClip = toSlMatrix(data.cameraViewToClip);
    constants.clipToCameraView = toSlMatrix(data.clipToCameraView);
    constants.mvecScale = {data.mvecScaleX, data.mvecScaleY};
    constants.jitterOffset = {data.jitterOffsetX, data.jitterOffsetY};
    constants.cameraPos = {data.cameraPos[0], data.cameraPos[1], data.cameraPos[2]};
    constants.cameraUp = {data.cameraUp[0], data.cameraUp[1], data.cameraUp[2]};
    constants.cameraRight = {data.cameraRight[0], data.cameraRight[1], data.cameraRight[2]};
    constants.cameraFwd = {data.cameraForward[0], data.cameraForward[1], data.cameraForward[2]};
    constants.cameraNear = data.cameraNear;
    constants.cameraFar = data.cameraFar;
    constants.cameraFOV = data.cameraFov;
    constants.cameraAspectRatio = data.cameraAspectRatio;
    constants.depthInverted = data.depthInverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = data.motionVectors3D ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = data.motionVectorsJittered ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.reset = (data.reset || m_needsReset) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.cameraPinholeOffset = {0.0f, 0.0f};
    constants.clipToPrevClip = toSlMatrix(data.clipToPrevClip);
    constants.prevClipToClip = toSlMatrix(data.prevClipToClip);

    m_needsReset = false;

    result = slSetConstants(constants, *frameToken, viewport);
    if (result != sl::Result::eOk) {
        spdlog::error("slSetConstants failed (result={})", static_cast<int>(result));
        return false;
    }

    // Tag resources
    sl::Resource colorIn = makeSlTextureResource(data.colorInput,
                                                 data.colorInputView,
                                                 data.colorInputInfo);
    sl::Resource colorOut = makeSlTextureResource(data.colorOutput,
                                                  data.colorOutputView,
                                                  data.colorOutputInfo);
    sl::Resource depthRes = makeSlTextureResource(data.depth,
                                                  data.depthView,
                                                  data.depthInfo);
    sl::Resource mvecRes = makeSlTextureResource(data.motionVectors,
                                                 data.motionVectorsView,
                                                 data.motionVectorsInfo);

    sl::Extent renderExtent{};
    renderExtent.left = 0;
    renderExtent.top = 0;
    renderExtent.width = data.renderWidth;
    renderExtent.height = data.renderHeight;

    sl::Extent displayExtent{};
    displayExtent.left = 0;
    displayExtent.top = 0;
    displayExtent.width = data.displayWidth;
    displayExtent.height = data.displayHeight;

    sl::ResourceTag tags[] = {
        sl::ResourceTag(&colorIn,  sl::kBufferTypeScalingInputColor,  sl::ResourceLifecycle::eValidUntilPresent, &renderExtent),
        sl::ResourceTag(&colorOut, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilPresent, &displayExtent),
        sl::ResourceTag(&depthRes, sl::kBufferTypeDepth,              sl::ResourceLifecycle::eValidUntilPresent, &renderExtent),
        sl::ResourceTag(&mvecRes,  sl::kBufferTypeMotionVectors,      sl::ResourceLifecycle::eValidUntilPresent, &renderExtent),
    };

    result = slSetTagForFrame(*frameToken, viewport, tags, _countof(tags),
                              static_cast<sl::CommandBuffer*>(data.commandBuffer));
    if (result != sl::Result::eOk) {
        spdlog::error("slSetTagForFrame failed (result={})", static_cast<int>(result));
        return false;
    }

    // Evaluate DLSS
    const sl::BaseStructure* inputs[] = {&viewport};
    result = slEvaluateFeature(sl::kFeatureDLSS, *frameToken, inputs, 1,
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
        sl::ViewportHandle viewport{kDefaultViewportId};
        slFreeResources(sl::kFeatureDLSS, viewport);
    }

    slShutdown();
    m_initialized = false;
    m_vulkanSet = false;
    m_dlssAvailable = false;
    m_vkGetDeviceProcAddrProxy = nullptr;
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

bool StreamlineContext::queryVulkanRequirements(StreamlineVulkanRequirements&) const {
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
