#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"

#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <vector>

#ifndef METALLIC_HAS_STREAMLINE
#define METALLIC_HAS_STREAMLINE 0
#endif

#if METALLIC_HAS_STREAMLINE
#include <sl.h>
#include <sl_dlss_d.h>
#include <sl_helpers_vk.h>
#include <sl_matrix_helpers.h>
#endif

namespace metallic::render::vulkan {
namespace {

#if METALLIC_HAS_STREAMLINE

struct StreamlineState {
    bool initialized = false;
    bool vulkanDeviceSet = false;
    bool dlssRrSupported = false;
    uint32_t frameIndex = 0;
    sl::ViewportHandle viewport{0};
};

std::mutex& streamlineMutex()
{
    static std::mutex mutex;
    return mutex;
}

StreamlineState& streamlineState()
{
    static StreamlineState state;
    return state;
}

template <typename Handle>
void* nativeHandleToVoid(Handle handle)
{
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<void*>(handle);
    } else {
        return reinterpret_cast<void*>(static_cast<uintptr_t>(handle));
    }
}

const char* slResultName(sl::Result result)
{
    switch (result) {
    case sl::Result::eOk:
        return "eOk";
    case sl::Result::eErrorIO:
        return "eErrorIO";
    case sl::Result::eErrorDriverOutOfDate:
        return "eErrorDriverOutOfDate";
    case sl::Result::eErrorOSOutOfDate:
        return "eErrorOSOutOfDate";
    case sl::Result::eErrorOSDisabledHWS:
        return "eErrorOSDisabledHWS";
    case sl::Result::eErrorDeviceNotCreated:
        return "eErrorDeviceNotCreated";
    case sl::Result::eErrorNoSupportedAdapterFound:
        return "eErrorNoSupportedAdapterFound";
    case sl::Result::eErrorAdapterNotSupported:
        return "eErrorAdapterNotSupported";
    case sl::Result::eErrorNoPlugins:
        return "eErrorNoPlugins";
    case sl::Result::eErrorVulkanAPI:
        return "eErrorVulkanAPI";
    case sl::Result::eErrorDXGIAPI:
        return "eErrorDXGIAPI";
    case sl::Result::eErrorD3DAPI:
        return "eErrorD3DAPI";
    case sl::Result::eErrorNRDAPI:
        return "eErrorNRDAPI";
    case sl::Result::eErrorNVAPI:
        return "eErrorNVAPI";
    case sl::Result::eErrorReflexAPI:
        return "eErrorReflexAPI";
    case sl::Result::eErrorNGXFailed:
        return "eErrorNGXFailed";
    case sl::Result::eErrorJSONParsing:
        return "eErrorJSONParsing";
    case sl::Result::eErrorMissingProxy:
        return "eErrorMissingProxy";
    case sl::Result::eErrorMissingResourceState:
        return "eErrorMissingResourceState";
    case sl::Result::eErrorInvalidIntegration:
        return "eErrorInvalidIntegration";
    case sl::Result::eErrorMissingInputParameter:
        return "eErrorMissingInputParameter";
    case sl::Result::eErrorNotInitialized:
        return "eErrorNotInitialized";
    case sl::Result::eErrorComputeFailed:
        return "eErrorComputeFailed";
    case sl::Result::eErrorInitNotCalled:
        return "eErrorInitNotCalled";
    case sl::Result::eErrorExceptionHandler:
        return "eErrorExceptionHandler";
    case sl::Result::eErrorInvalidParameter:
        return "eErrorInvalidParameter";
    case sl::Result::eErrorMissingConstants:
        return "eErrorMissingConstants";
    case sl::Result::eErrorDuplicatedConstants:
        return "eErrorDuplicatedConstants";
    case sl::Result::eErrorMissingOrInvalidAPI:
        return "eErrorMissingOrInvalidAPI";
    case sl::Result::eErrorCommonConstantsMissing:
        return "eErrorCommonConstantsMissing";
    case sl::Result::eErrorUnsupportedInterface:
        return "eErrorUnsupportedInterface";
    case sl::Result::eErrorFeatureMissing:
        return "eErrorFeatureMissing";
    case sl::Result::eErrorFeatureNotSupported:
        return "eErrorFeatureNotSupported";
    case sl::Result::eErrorFeatureMissingHooks:
        return "eErrorFeatureMissingHooks";
    case sl::Result::eErrorFeatureFailedToLoad:
        return "eErrorFeatureFailedToLoad";
    case sl::Result::eErrorFeatureWrongPriority:
        return "eErrorFeatureWrongPriority";
    case sl::Result::eErrorFeatureMissingDependency:
        return "eErrorFeatureMissingDependency";
    case sl::Result::eErrorFeatureManagerInvalidState:
        return "eErrorFeatureManagerInvalidState";
    case sl::Result::eErrorInvalidState:
        return "eErrorInvalidState";
    case sl::Result::eWarnOutOfVRAM:
        return "eWarnOutOfVRAM";
    default:
        return "Streamline error";
    }
}

Error errorFromSl(sl::Result result)
{
    switch (result) {
    case sl::Result::eOk:
        return Error::Failure;
    case sl::Result::eWarnOutOfVRAM:
        return Error::OutOfMemory;
    case sl::Result::eErrorNoSupportedAdapterFound:
    case sl::Result::eErrorAdapterNotSupported:
    case sl::Result::eErrorNoPlugins:
    case sl::Result::eErrorFeatureMissing:
    case sl::Result::eErrorFeatureNotSupported:
    case sl::Result::eErrorFeatureFailedToLoad:
        return Error::Unsupported;
    case sl::Result::eErrorMissingInputParameter:
    case sl::Result::eErrorInvalidParameter:
        return Error::InvalidArgument;
    default:
        return Error::Failure;
    }
}

Result resultFromSl(sl::Result result, const char* label, std::string& log)
{
    if (result == sl::Result::eOk) {
        return {};
    }

    if (!log.empty() && log.back() != '\n') {
        log += '\n';
    }
    log += label;
    log += " returned ";
    log += slResultName(result);
    return makeError(errorFromSl(result));
}

bool isUnsupportedStateTrackingHookWarning(const char* message)
{
    return std::strstr(message, "Hook sl.common:Vulkan:CmdBindPipeline is NOT supported") != nullptr ||
        std::strstr(message, "Hook sl.common:Vulkan:CmdBindDescriptorSets is NOT supported") != nullptr ||
        std::strstr(message, "Hook sl.common:Vulkan:BeginCommandBuffer is NOT supported") != nullptr;
}

void streamlineLogCallback(sl::LogType type, const char* message)
{
    if (message == nullptr) {
        return;
    }

    const char* level = "info";
    if (type == sl::LogType::eWarn && isUnsupportedStateTrackingHookWarning(message)) {
        return;
    }
    if (type == sl::LogType::eWarn) {
        level = "warn";
    } else if (type == sl::LogType::eError) {
        level = "error";
    }
    std::cerr << "[Streamline][" << level << "] " << message << '\n';
}

sl::DLSSMode slMode(StreamlineDlssRrMode mode)
{
    switch (mode) {
    case StreamlineDlssRrMode::Dlaa:
        return sl::DLSSMode::eDLAA;
    case StreamlineDlssRrMode::Quality:
        return sl::DLSSMode::eMaxQuality;
    case StreamlineDlssRrMode::Balanced:
        return sl::DLSSMode::eBalanced;
    case StreamlineDlssRrMode::Performance:
        return sl::DLSSMode::eMaxPerformance;
    case StreamlineDlssRrMode::UltraPerformance:
        return sl::DLSSMode::eUltraPerformance;
    case StreamlineDlssRrMode::UltraQuality:
        return sl::DLSSMode::eUltraQuality;
    case StreamlineDlssRrMode::Off:
        return sl::DLSSMode::eOff;
    }
    return sl::DLSSMode::eBalanced;
}

void setIdentity(sl::float4x4& matrix)
{
    matrix[0] = sl::float4(1.0f, 0.0f, 0.0f, 0.0f);
    matrix[1] = sl::float4(0.0f, 1.0f, 0.0f, 0.0f);
    matrix[2] = sl::float4(0.0f, 0.0f, 1.0f, 0.0f);
    matrix[3] = sl::float4(0.0f, 0.0f, 0.0f, 1.0f);
}

float dot3(sl::float3 a, sl::float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

sl::float3 cross3(sl::float3 a, sl::float3 b)
{
    return sl::float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

sl::float3 subtract3(sl::float3 a, sl::float3 b)
{
    return sl::float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

sl::float3 normalizeOr(sl::float3 value, sl::float3 fallback)
{
    const float lengthSquared = dot3(value, value);
    if (lengthSquared <= 0.00000001f) {
        return fallback;
    }
    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
    return sl::float3(value.x * inverseLength, value.y * inverseLength, value.z * inverseLength);
}

sl::float3 cameraArrayToFloat3(const float values[3])
{
    return sl::float3(values[0], values[1], values[2]);
}

void cameraBasis(
    const StreamlineDlssRrCamera& camera,
    bool previous,
    sl::float3& outPosition,
    sl::float3& outRight,
    sl::float3& outUp,
    sl::float3& outForward)
{
    outPosition = cameraArrayToFloat3(previous ? camera.previousEye : camera.eye);
    const sl::float3 center = cameraArrayToFloat3(previous ? camera.previousCenter : camera.center);
    const sl::float3 upInput = cameraArrayToFloat3(previous ? camera.previousUp : camera.up);
    outForward = normalizeOr(subtract3(center, outPosition), sl::float3(0.0f, 0.0f, -1.0f));
    outRight = normalizeOr(cross3(outForward, upInput), sl::float3(1.0f, 0.0f, 0.0f));
    outUp = normalizeOr(cross3(outRight, outForward), sl::float3(0.0f, 1.0f, 0.0f));
}

sl::float4x4 makeCameraViewToWorld(
    sl::float3 position,
    sl::float3 right,
    sl::float3 up,
    sl::float3 forward)
{
    sl::float4x4 matrix;
    matrix[0] = sl::float4(right.x, right.y, right.z, 0.0f);
    matrix[1] = sl::float4(up.x, up.y, up.z, 0.0f);
    matrix[2] = sl::float4(forward.x, forward.y, forward.z, 0.0f);
    matrix[3] = sl::float4(position.x, position.y, position.z, 1.0f);
    return matrix;
}

sl::float4x4 makeCameraViewToClip(
    float fovRadians,
    float aspectRatio,
    float zNear,
    float zFar,
    float orthoHeight,
    bool orthographic)
{
    sl::float4x4 matrix;
    setIdentity(matrix);
    const float safeAspect = std::max(aspectRatio, 0.001f);
    const float safeNear = std::max(zNear, 0.0001f);
    const float safeFar = std::max(zFar, safeNear + 0.001f);
    if (orthographic) {
        const float height = std::max(orthoHeight, 0.0001f);
        const float width = std::max(height * safeAspect, 0.0001f);
        matrix[0] = sl::float4(2.0f / width, 0.0f, 0.0f, 0.0f);
        matrix[1] = sl::float4(0.0f, 2.0f / height, 0.0f, 0.0f);
        matrix[2] = sl::float4(0.0f, 0.0f, 1.0f / (safeFar - safeNear), 0.0f);
        matrix[3] = sl::float4(0.0f, 0.0f, -safeNear / (safeFar - safeNear), 1.0f);
        return matrix;
    }

    const float safeFov = std::clamp(fovRadians, 0.017453292f, 3.12413936f);
    const float yScale = 1.0f / std::tan(safeFov * 0.5f);
    const float xScale = yScale / safeAspect;
    matrix[0] = sl::float4(xScale, 0.0f, 0.0f, 0.0f);
    matrix[1] = sl::float4(0.0f, yScale, 0.0f, 0.0f);
    matrix[2] = sl::float4(0.0f, 0.0f, safeFar / (safeFar - safeNear), 1.0f);
    matrix[3] = sl::float4(0.0f, 0.0f, -(safeNear * safeFar) / (safeFar - safeNear), 0.0f);
    return matrix;
}

struct SlCameraMatrices {
    sl::float4x4 cameraViewToWorld;
    sl::float4x4 worldToCameraView;
    sl::float4x4 cameraViewToClip;
    sl::float4x4 clipToCameraView;
    sl::float3 position;
    sl::float3 right;
    sl::float3 up;
    sl::float3 forward;
    float fovRadians = 0.87266463f;
    float aspectRatio = 1.0f;
    float zNear = 0.001f;
    float zFar = 10000.0f;
    bool orthographic = false;
};

SlCameraMatrices makeCameraMatrices(const StreamlineDlssRrCamera& camera, bool previous)
{
    SlCameraMatrices matrices;
    cameraBasis(camera, previous, matrices.position, matrices.right, matrices.up, matrices.forward);
    matrices.fovRadians = previous ? camera.previousFovRadians : camera.fovRadians;
    matrices.aspectRatio = previous ? camera.previousAspectRatio : camera.aspectRatio;
    matrices.zNear = previous ? camera.previousZNear : camera.zNear;
    matrices.zFar = previous ? camera.previousZFar : camera.zFar;
    const float orthoHeight = previous ? camera.previousOrthoHeight : camera.orthoHeight;
    matrices.orthographic = previous ? camera.previousOrthographic : camera.orthographic;
    matrices.cameraViewToWorld = makeCameraViewToWorld(
        matrices.position,
        matrices.right,
        matrices.up,
        matrices.forward);
    sl::matrixOrthoNormalInvert(matrices.worldToCameraView, matrices.cameraViewToWorld);
    matrices.cameraViewToClip = makeCameraViewToClip(
        matrices.fovRadians,
        matrices.aspectRatio,
        matrices.zNear,
        matrices.zFar,
        orthoHeight,
        matrices.orthographic);
    sl::matrixFullInvert(matrices.clipToCameraView, matrices.cameraViewToClip);
    return matrices;
}

sl::Constants makeConstants(const StreamlineDlssRrDesc& desc)
{
    const SlCameraMatrices currentCamera = makeCameraMatrices(desc.camera, false);
    const SlCameraMatrices previousCamera = desc.camera.previousValid
        ? makeCameraMatrices(desc.camera, true)
        : currentCamera;

    sl::Constants constants;
    constants.cameraViewToClip = currentCamera.cameraViewToClip;
    constants.clipToCameraView = currentCamera.clipToCameraView;
    setIdentity(constants.clipToLensClip);
    sl::float4x4 cameraViewToPrevCameraView;
    sl::calcCameraToPrevCamera(
        cameraViewToPrevCameraView,
        currentCamera.cameraViewToWorld,
        previousCamera.cameraViewToWorld);
    sl::float4x4 clipToPrevCameraView;
    sl::matrixMul(clipToPrevCameraView, currentCamera.clipToCameraView, cameraViewToPrevCameraView);
    sl::matrixMul(constants.clipToPrevClip, clipToPrevCameraView, previousCamera.cameraViewToClip);
    sl::matrixFullInvert(constants.prevClipToClip, constants.clipToPrevClip);
    constants.jitterOffset = sl::float2(0.0f, 0.0f);
    constants.mvecScale = sl::float2(1.0f, 1.0f);
    constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
    constants.cameraPos = currentCamera.position;
    constants.cameraUp = currentCamera.up;
    constants.cameraRight = currentCamera.right;
    constants.cameraFwd = currentCamera.forward;
    constants.cameraNear = currentCamera.zNear;
    constants.cameraFar = currentCamera.zFar;
    constants.cameraFOV = currentCamera.fovRadians;
    constants.cameraAspectRatio = currentCamera.aspectRatio;
    constants.motionVectorsInvalidValue = 0.0f;
    constants.depthInverted = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.reset = (desc.reset || !desc.camera.previousValid) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.orthographicProjection = currentCamera.orthographic ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    return constants;
}

sl::DLSSDOptions makeDlssRrOptions(const StreamlineDlssRrDesc& desc)
{
    sl::DLSSDOptions options;
    options.mode = slMode(desc.mode);
    options.outputWidth = desc.width;
    options.outputHeight = desc.height;
    options.colorBuffersHDR = sl::Boolean::eTrue;
    options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
    options.alphaUpscalingEnabled = sl::Boolean::eFalse;
    const SlCameraMatrices currentCamera = makeCameraMatrices(desc.camera, false);
    options.worldToCameraView = currentCamera.worldToCameraView;
    options.cameraViewToWorld = currentCamera.cameraViewToWorld;
    return options;
}

bool validTextureRef(const StreamlineDlssRrTextureRef& ref)
{
    return ref.texture != nullptr && ref.view != nullptr;
}

bool appendResourceTag(
    const StreamlineDlssRrTextureRef& ref,
    sl::BufferType type,
    const sl::Extent& extent,
    std::vector<sl::Resource>& resources,
    std::vector<sl::ResourceTag>& tags)
{
    if (!validTextureRef(ref)) {
        return false;
    }

    const NativeTexture texture = nativeTexture(*ref.texture);
    const VkImageView imageView = nativeImageView(*ref.view);
    if (texture.image == VK_NULL_HANDLE || imageView == VK_NULL_HANDLE) {
        return false;
    }

    resources.push_back(sl::Resource(
        sl::ResourceType::eTex2d,
        nativeHandleToVoid(texture.image),
        nativeHandleToVoid(texture.memory),
        nativeHandleToVoid(imageView),
        static_cast<uint32_t>(VK_IMAGE_LAYOUT_GENERAL)));
    sl::Resource& resource = resources.back();
    resource.width = texture.width;
    resource.height = texture.height;
    resource.nativeFormat = static_cast<uint32_t>(texture.format);
    resource.mipLevels = texture.mipCount;
    resource.arrayLayers = texture.layerCount;
    resource.flags = static_cast<uint32_t>(texture.flags);
    resource.usage = static_cast<uint32_t>(texture.usage);
    tags.push_back(sl::ResourceTag(&resource, type, sl::eValidUntilEvaluate, &extent));
    return true;
}

sl::CommandBuffer* slCommandBuffer(CommandBuffer& commandBuffer)
{
    return static_cast<sl::CommandBuffer*>(nativeHandleToVoid(nativeCommandBuffer(commandBuffer)));
}

#endif

} // namespace

const char* streamlineVulkanLibraryName()
{
#if METALLIC_HAS_STREAMLINE
    return METALLIC_STREAMLINE_INTERPOSER_DLL;
#else
    return nullptr;
#endif
}

bool streamlineSdkAvailable()
{
#if METALLIC_HAS_STREAMLINE
    return true;
#else
    return false;
#endif
}

bool streamlineInitialized()
{
#if METALLIC_HAS_STREAMLINE
    std::lock_guard lock(streamlineMutex());
    return streamlineState().initialized;
#else
    return false;
#endif
}

bool streamlineDlssRrSupported()
{
#if METALLIC_HAS_STREAMLINE
    std::lock_guard lock(streamlineMutex());
    return streamlineState().dlssRrSupported;
#else
    return false;
#endif
}

Result initializeStreamlinePreDevice(std::string& log)
{
#if !METALLIC_HAS_STREAMLINE
    log = "NVIDIA Streamline SDK is not available";
    return makeError(Error::Unsupported);
#else
    std::lock_guard lock(streamlineMutex());
    StreamlineState& state = streamlineState();
    if (state.initialized) {
        return {};
    }

    const std::wstring pluginPath = std::filesystem::path(METALLIC_STREAMLINE_BIN_DIR).wstring();
    const wchar_t* pluginPaths[] = {pluginPath.c_str()};
    const sl::Feature features[] = {
        sl::kFeatureDLSS_RR,
        sl::kFeaturePCL,
    };

    sl::Preferences preferences;
    preferences.renderAPI = sl::RenderAPI::eVulkan;
    preferences.engine = sl::EngineType::eCustom;
    // Leave projectId/engineVersion empty so NGX uses the registered applicationId.
    preferences.applicationId = 231313132;
    preferences.pathsToPlugins = pluginPaths;
    preferences.numPathsToPlugins = static_cast<uint32_t>(std::size(pluginPaths));
    preferences.featuresToLoad = features;
    preferences.numFeaturesToLoad = static_cast<uint32_t>(std::size(features));
    preferences.logLevel = sl::LogLevel::eDefault;
    preferences.logMessageCallback = streamlineLogCallback;
    preferences.flags =
        sl::PreferenceFlags::eDisableCLStateTracking |
        sl::PreferenceFlags::eLoadDownloadedPlugins |
        sl::PreferenceFlags::eDisableDebugText |
        sl::PreferenceFlags::eUseManualHooking |
        sl::PreferenceFlags::eUseFrameBasedResourceTagging;

    Result result = resultFromSl(slInit(preferences, sl::kSDKVersion), "slInit", log);
    if (!result) {
        return result;
    }

    state = StreamlineState{};
    state.initialized = true;
    return {};
#endif
}

Result setStreamlineVulkanDevice(
    const NativeDevice& device,
    const NativeQueue& graphicsQueue,
    const NativeQueue& computeQueue,
    std::string& log)
{
#if !METALLIC_HAS_STREAMLINE
    (void)device;
    (void)graphicsQueue;
    (void)computeQueue;
    log = "NVIDIA Streamline SDK is not available";
    return makeError(Error::Unsupported);
#else
    std::lock_guard lock(streamlineMutex());
    StreamlineState& state = streamlineState();
    if (!state.initialized) {
        log = "Streamline was not initialized before Vulkan device creation";
        return makeError(Error::InvalidArgument);
    }

    if (device.instance == VK_NULL_HANDLE ||
        device.physicalDevice == VK_NULL_HANDLE ||
        device.device == VK_NULL_HANDLE ||
        graphicsQueue.queue == VK_NULL_HANDLE ||
        computeQueue.queue == VK_NULL_HANDLE) {
        log = "Streamline Vulkan setup received invalid native handles";
        return makeError(Error::InvalidArgument);
    }

    state.vulkanDeviceSet = true;

    sl::AdapterInfo adapterInfo;
    adapterInfo.vkPhysicalDevice = nativeHandleToVoid(device.physicalDevice);
    const sl::Result supportResult = slIsFeatureSupported(sl::kFeatureDLSS_RR, adapterInfo);
    state.dlssRrSupported = supportResult == sl::Result::eOk;
    if (!state.dlssRrSupported) {
        if (!log.empty() && log.back() != '\n') {
            log += '\n';
        }
        log += "slIsFeatureSupported(kFeatureDLSS_RR) returned ";
        log += slResultName(supportResult);
    }
    return {};
#endif
}

void shutdownStreamline()
{
#if METALLIC_HAS_STREAMLINE
    std::lock_guard lock(streamlineMutex());
    StreamlineState& state = streamlineState();
    if (!state.initialized) {
        return;
    }

    if (state.vulkanDeviceSet && state.dlssRrSupported) {
        (void)slFreeResources(sl::kFeatureDLSS_RR, state.viewport);
    }
    (void)slShutdown();
    state = StreamlineState{};
#endif
}

Result evaluateStreamlineDlssRr(CommandBuffer& commandBuffer, const StreamlineDlssRrDesc& desc, std::string& log)
{
#if !METALLIC_HAS_STREAMLINE
    (void)commandBuffer;
    (void)desc;
    log = "NVIDIA Streamline SDK is not available";
    return makeError(Error::Unsupported);
#else
    std::lock_guard lock(streamlineMutex());
    StreamlineState& state = streamlineState();
    if (!state.initialized || !state.vulkanDeviceSet || !state.dlssRrSupported) {
        log = "DLSS-RR is not available on the current Vulkan device";
        return makeError(Error::Unsupported);
    }
    if (desc.width == 0 || desc.height == 0 || desc.mode == StreamlineDlssRrMode::Off) {
        return makeError(Error::InvalidArgument);
    }

    sl::CommandBuffer* nativeCommandBuffer = slCommandBuffer(commandBuffer);
    if (nativeCommandBuffer == nullptr) {
        log = "DLSS-RR evaluate received an invalid command buffer";
        return makeError(Error::InvalidArgument);
    }

    sl::FrameToken* frameToken = nullptr;
    const uint32_t frameIndex = state.frameIndex++;
    Result result = resultFromSl(slGetNewFrameToken(frameToken, &frameIndex), "slGetNewFrameToken", log);
    if (!result) {
        return result;
    }
    if (frameToken == nullptr) {
        log = "slGetNewFrameToken returned a null token";
        return makeError(Error::Failure);
    }

    sl::DLSSDOptions options = makeDlssRrOptions(desc);
    result = resultFromSl(slDLSSDSetOptions(state.viewport, options), "slDLSSDSetOptions", log);
    if (!result) {
        return result;
    }

    sl::Constants constants = makeConstants(desc);
    result = resultFromSl(slSetConstants(constants, *frameToken, state.viewport), "slSetConstants", log);
    if (!result) {
        return result;
    }

    sl::Extent extent{
        .top = 0,
        .left = 0,
        .width = desc.width,
        .height = desc.height,
    };
    std::vector<sl::Resource> resources;
    std::vector<sl::ResourceTag> tags;
    resources.reserve(8);
    tags.reserve(8);
    const bool validResources =
        appendResourceTag(desc.inputColor, sl::kBufferTypeScalingInputColor, extent, resources, tags) &&
        appendResourceTag(desc.outputColor, sl::kBufferTypeScalingOutputColor, extent, resources, tags) &&
        appendResourceTag(desc.albedo, sl::kBufferTypeAlbedo, extent, resources, tags) &&
        appendResourceTag(desc.specularAlbedo, sl::kBufferTypeSpecularAlbedo, extent, resources, tags) &&
        appendResourceTag(desc.normalRoughness, sl::kBufferTypeNormalRoughness, extent, resources, tags) &&
        appendResourceTag(desc.motionVectors, sl::kBufferTypeMotionVectors, extent, resources, tags) &&
        appendResourceTag(desc.linearDepth, sl::kBufferTypeLinearDepth, extent, resources, tags) &&
        appendResourceTag(desc.specularHitDistance, sl::kBufferTypeSpecularHitDistance, extent, resources, tags);
    if (!validResources) {
        log = "DLSS-RR evaluate received invalid texture resources";
        return makeError(Error::InvalidArgument);
    }

    result = resultFromSl(
        slSetTagForFrame(*frameToken, state.viewport, tags.data(), static_cast<uint32_t>(tags.size()), nativeCommandBuffer),
        "slSetTagForFrame",
        log);
    if (!result) {
        return result;
    }

    const sl::BaseStructure* inputs[] = {&state.viewport};
    return resultFromSl(
        slEvaluateFeature(
            sl::kFeatureDLSS_RR,
            *frameToken,
            inputs,
            static_cast<uint32_t>(std::size(inputs)),
            nativeCommandBuffer),
        "slEvaluateFeature(kFeatureDLSS_RR)",
        log);
#endif
}

} // namespace metallic::render::vulkan
