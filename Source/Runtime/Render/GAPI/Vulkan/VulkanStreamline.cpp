#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"

#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <vector>

#ifndef METALLIC_HAS_STREAMLINE
#define METALLIC_HAS_STREAMLINE 0
#endif

#if METALLIC_HAS_STREAMLINE
#include <sl.h>
#include <sl_dlss.h>
#include <sl_dlss_d.h>
#include <sl_helpers_vk.h>
#include <sl_matrix_helpers.h>
#endif

namespace metallic::render::vulkan {
namespace {

#if METALLIC_HAS_STREAMLINE

struct DlssRrOptimalSettingsCacheEntry {
    StreamlineDlssRrMode mode = StreamlineDlssRrMode::Off;
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    StreamlineDlssRrOptimalSettings settings;
};

struct StreamlineState {
    bool initialized = false;
    bool vulkanDeviceSet = false;
    bool dlssSrSupported = false;
    bool dlssRrSupported = false;
    bool descriptorHeapWorkaroundEnabled = false;
    uint32_t frameIndex = 0;
    sl::ViewportHandle viewport{0};
    VkDevice vulkanDevice = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::vector<DlssRrOptimalSettingsCacheEntry> dlssSrOptimalSettingsCache;
    std::vector<DlssRrOptimalSettingsCacheEntry> dlssRrOptimalSettingsCache;
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

// VK_EXT_descriptor_heap heap commands and legacy descriptor-set commands invalidate
// each other's state. Streamline 2.11.1 still records descriptor-set based DLSS work,
// and NVIDIA-RTX/Streamline#109 reports black output when a heap remains active.
// Bind an empty legacy set immediately before evaluate, then tell the RHI to rebind
// its heap before the next engine dispatch.
void destroyDescriptorHeapWorkaround(StreamlineState& state)
{
    if (state.vulkanDevice == VK_NULL_HANDLE) {
        return;
    }
    if (state.descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(state.vulkanDevice, state.descriptorPool, nullptr);
        state.descriptorPool = VK_NULL_HANDLE;
        state.descriptorSet = VK_NULL_HANDLE;
    }
    if (state.pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(state.vulkanDevice, state.pipelineLayout, nullptr);
        state.pipelineLayout = VK_NULL_HANDLE;
    }
    if (state.descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(state.vulkanDevice, state.descriptorSetLayout, nullptr);
        state.descriptorSetLayout = VK_NULL_HANDLE;
    }
    state.descriptorHeapWorkaroundEnabled = false;
}

Result initializeDescriptorHeapWorkaround(
    StreamlineState& state,
    const NativeDevice& device,
    std::string& log)
{
    state.vulkanDevice = device.device;
    if (!device.descriptorHeapEnabled) {
        return {};
    }

    const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    };
    VkResult vkResult = vkCreateDescriptorSetLayout(
        device.device,
        &setLayoutInfo,
        nullptr,
        &state.descriptorSetLayout);
    if (vkResult == VK_SUCCESS) {
        const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &state.descriptorSetLayout,
        };
        vkResult = vkCreatePipelineLayout(
            device.device,
            &pipelineLayoutInfo,
            nullptr,
            &state.pipelineLayout);
    }
    if (vkResult == VK_SUCCESS) {
        const VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
        };
        vkResult = vkCreateDescriptorPool(
            device.device,
            &poolInfo,
            nullptr,
            &state.descriptorPool);
    }
    if (vkResult == VK_SUCCESS) {
        const VkDescriptorSetAllocateInfo allocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = state.descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &state.descriptorSetLayout,
        };
        vkResult = vkAllocateDescriptorSets(device.device, &allocateInfo, &state.descriptorSet);
    }
    if (vkResult != VK_SUCCESS) {
        log = "Failed to initialize the Streamline VK_EXT_descriptor_heap workaround (VkResult " +
            std::to_string(static_cast<int32_t>(vkResult)) + ")";
        destroyDescriptorHeapWorkaround(state);
        return makeError(Error::Failure);
    }

    state.descriptorHeapWorkaroundEnabled = true;
    spdlog::info(
        "[Streamline] Enabled VK_EXT_descriptor_heap compatibility workaround before feature evaluation");
    return {};
}

void prepareDescriptorStateForStreamline(
    StreamlineState& state,
    CommandBuffer& commandBuffer)
{
    if (!state.descriptorHeapWorkaroundEnabled) {
        return;
    }

    const VkCommandBuffer commandBufferHandle = nativeCommandBuffer(commandBuffer);
    vkCmdBindDescriptorSets(
        commandBufferHandle,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        state.pipelineLayout,
        0,
        1,
        &state.descriptorSet,
        0,
        nullptr);
    notifyExternalDescriptorSetBinding(commandBuffer);
}

void streamlineLogCallback(sl::LogType type, const char* message)
{
    if (message == nullptr) {
        return;
    }

    if (type == sl::LogType::eInfo) {
        return;
    }
    if (type == sl::LogType::eWarn && isUnsupportedStateTrackingHookWarning(message)) {
        return;
    }
    if (type != sl::LogType::eWarn && type != sl::LogType::eError) {
        return;
    }

    const char* level = type == sl::LogType::eWarn ? "warn" : "error";
    spdlog::log(
        type == sl::LogType::eWarn ? spdlog::level::warn : spdlog::level::err,
        "[Streamline][{}] {}",
        level,
        message);
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

sl::Constants makeConstants(const StreamlineDlssRrCamera& camera, bool reset)
{
    const SlCameraMatrices currentCamera = makeCameraMatrices(camera, false);
    const SlCameraMatrices previousCamera = camera.previousValid
        ? makeCameraMatrices(camera, true)
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
    constants.jitterOffset = sl::float2(camera.jitterOffset[0], camera.jitterOffset[1]);
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
    constants.reset = (reset || !camera.previousValid) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.orthographicProjection = currentCamera.orthographic ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    return constants;
}

sl::DLSSOptions makeDlssSrBaseOptions(
    StreamlineDlssSrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight)
{
    sl::DLSSOptions options;
    options.mode = slMode(mode);
    options.outputWidth = outputWidth;
    options.outputHeight = outputHeight;
    options.colorBuffersHDR = sl::Boolean::eTrue;
    options.useAutoExposure = sl::Boolean::eTrue;
    options.alphaUpscalingEnabled = sl::Boolean::eFalse;
    return options;
}

sl::DLSSDOptions makeDlssRrBaseOptions(
    StreamlineDlssRrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight)
{
    sl::DLSSDOptions options;
    options.mode = slMode(mode);
    options.outputWidth = outputWidth;
    options.outputHeight = outputHeight;
    options.sharpness = 0.0f;
    options.preExposure = 1.0f;
    options.exposureScale = 1.0f;
    options.colorBuffersHDR = sl::Boolean::eTrue;
    options.indicatorInvertAxisX = sl::Boolean::eFalse;
    options.indicatorInvertAxisY = sl::Boolean::eFalse;
    options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
    options.alphaUpscalingEnabled = sl::Boolean::eFalse;
    options.dlaaPreset = sl::DLSSDPreset::ePresetD;
    options.qualityPreset = sl::DLSSDPreset::ePresetD;
    options.balancedPreset = sl::DLSSDPreset::ePresetD;
    options.performancePreset = sl::DLSSDPreset::ePresetD;
    options.ultraPerformancePreset = sl::DLSSDPreset::ePresetD;
    options.ultraQualityPreset = sl::DLSSDPreset::ePresetD;
    return options;
}

sl::DLSSDOptions makeDlssRrOptions(const StreamlineDlssRrDesc& desc)
{
    sl::DLSSDOptions options = makeDlssRrBaseOptions(
        desc.mode,
        desc.outputWidth,
        desc.outputHeight);
    const SlCameraMatrices currentCamera = makeCameraMatrices(desc.camera, false);
    options.worldToCameraView = currentCamera.worldToCameraView;
    options.cameraViewToWorld = currentCamera.cameraViewToWorld;
    return options;
}

bool validTextureRef(const StreamlineDlssRrTextureRef& ref)
{
    return ref.texture != nullptr && ref.view != nullptr;
}

bool textureRefMatchesExtent(
    const StreamlineDlssRrTextureRef& ref,
    uint32_t width,
    uint32_t height)
{
    return validTextureRef(ref) &&
        ref.texture->desc().width == width &&
        ref.texture->desc().height == height;
}

bool appendResourceTag(
    const StreamlineDlssRrTextureRef& ref,
    sl::BufferType type,
    const sl::Extent& extent,
    std::vector<sl::Resource>& resources,
    std::vector<sl::SubresourceRange>& subresourceRanges,
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
        nullptr,
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

    subresourceRanges.emplace_back();
    sl::SubresourceRange& subresourceRange = subresourceRanges.back();
    subresourceRange.aspectMask = type == sl::kBufferTypeDepth
        ? static_cast<uint32_t>(VK_IMAGE_ASPECT_DEPTH_BIT)
        : static_cast<uint32_t>(VK_IMAGE_ASPECT_COLOR_BIT);
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = texture.mipCount;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = texture.layerCount;
    resource.next = &subresourceRange;

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

bool streamlineDlssSrSupported()
{
#if METALLIC_HAS_STREAMLINE
    std::lock_guard lock(streamlineMutex());
    return streamlineState().dlssSrSupported;
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

Result getStreamlineDlssSrOptimalSettings(
    StreamlineDlssSrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight,
    StreamlineDlssSrOptimalSettings& settings,
    std::string& log)
{
    settings = {};
#if !METALLIC_HAS_STREAMLINE
    (void)mode;
    (void)outputWidth;
    (void)outputHeight;
    log = "NVIDIA Streamline SDK is not available";
    return makeError(Error::Unsupported);
#else
    std::lock_guard lock(streamlineMutex());
    StreamlineState& state = streamlineState();
    if (!state.initialized || !state.vulkanDeviceSet || !state.dlssSrSupported) {
        log = "DLSS-SR optimal settings are not available on the current Vulkan device";
        return makeError(Error::Unsupported);
    }
    if (mode == StreamlineDlssSrMode::Off || outputWidth == 0 || outputHeight == 0) {
        log = "DLSS-SR optimal settings require an enabled mode and non-zero output dimensions";
        return makeError(Error::InvalidArgument);
    }

    const auto cached = std::find_if(
        state.dlssSrOptimalSettingsCache.begin(),
        state.dlssSrOptimalSettingsCache.end(),
        [mode, outputWidth, outputHeight](const DlssRrOptimalSettingsCacheEntry& entry) {
            return entry.mode == mode &&
                entry.outputWidth == outputWidth &&
                entry.outputHeight == outputHeight;
        });
    if (cached != state.dlssSrOptimalSettingsCache.end()) {
        settings = cached->settings;
        return {};
    }

    sl::DLSSOptions options = makeDlssSrBaseOptions(mode, outputWidth, outputHeight);
    sl::DLSSOptimalSettings nativeSettings;
    Result result = resultFromSl(
        slDLSSGetOptimalSettings(options, nativeSettings),
        "slDLSSGetOptimalSettings",
        log);
    if (!result) {
        return result;
    }

    settings = StreamlineDlssSrOptimalSettings{
        .renderWidth = nativeSettings.optimalRenderWidth,
        .renderHeight = nativeSettings.optimalRenderHeight,
        .renderWidthMin = nativeSettings.renderWidthMin,
        .renderHeightMin = nativeSettings.renderHeightMin,
        .renderWidthMax = nativeSettings.renderWidthMax,
        .renderHeightMax = nativeSettings.renderHeightMax,
    };
    const bool invalidRanges =
        (settings.renderWidthMin != 0 && settings.renderWidthMax != 0 &&
            settings.renderWidthMin > settings.renderWidthMax) ||
        (settings.renderHeightMin != 0 && settings.renderHeightMax != 0 &&
            settings.renderHeightMin > settings.renderHeightMax);
    const bool outsideReportedRange =
        (settings.renderWidthMin != 0 && settings.renderWidth < settings.renderWidthMin) ||
        (settings.renderHeightMin != 0 && settings.renderHeight < settings.renderHeightMin) ||
        (settings.renderWidthMax != 0 && settings.renderWidth > settings.renderWidthMax) ||
        (settings.renderHeightMax != 0 && settings.renderHeight > settings.renderHeightMax);
    if (settings.renderWidth == 0 ||
        settings.renderHeight == 0 ||
        settings.renderWidth > outputWidth ||
        settings.renderHeight > outputHeight ||
        invalidRanges ||
        outsideReportedRange) {
        log = "slDLSSGetOptimalSettings returned invalid render dimensions " +
            std::to_string(settings.renderWidth) + "x" + std::to_string(settings.renderHeight) +
            " for output " + std::to_string(outputWidth) + "x" + std::to_string(outputHeight);
        settings = {};
        return makeError(Error::InvalidArgument);
    }

    state.dlssSrOptimalSettingsCache.push_back(DlssRrOptimalSettingsCacheEntry{
        .mode = mode,
        .outputWidth = outputWidth,
        .outputHeight = outputHeight,
        .settings = settings,
    });
    spdlog::info(
        "[Streamline] DLSS-SR optimal render extent {}x{} for output {}x{}",
        settings.renderWidth,
        settings.renderHeight,
        outputWidth,
        outputHeight);
    return {};
#endif
}

Result getStreamlineDlssRrOptimalSettings(
    StreamlineDlssRrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight,
    StreamlineDlssRrOptimalSettings& settings,
    std::string& log)
{
    settings = {};
#if !METALLIC_HAS_STREAMLINE
    (void)mode;
    (void)outputWidth;
    (void)outputHeight;
    log = "NVIDIA Streamline SDK is not available";
    return makeError(Error::Unsupported);
#else
    std::lock_guard lock(streamlineMutex());
    StreamlineState& state = streamlineState();
    if (!state.initialized || !state.vulkanDeviceSet || !state.dlssRrSupported) {
        log = "DLSS-RR optimal settings are not available on the current Vulkan device";
        return makeError(Error::Unsupported);
    }
    if (mode == StreamlineDlssRrMode::Off || outputWidth == 0 || outputHeight == 0) {
        log = "DLSS-RR optimal settings require an enabled mode and non-zero output dimensions";
        return makeError(Error::InvalidArgument);
    }

    const auto cached = std::find_if(
        state.dlssRrOptimalSettingsCache.begin(),
        state.dlssRrOptimalSettingsCache.end(),
        [mode, outputWidth, outputHeight](const DlssRrOptimalSettingsCacheEntry& entry) {
            return entry.mode == mode &&
                entry.outputWidth == outputWidth &&
                entry.outputHeight == outputHeight;
        });
    if (cached != state.dlssRrOptimalSettingsCache.end()) {
        settings = cached->settings;
        return {};
    }

    sl::DLSSDOptions options = makeDlssRrBaseOptions(mode, outputWidth, outputHeight);
    sl::DLSSDOptimalSettings nativeSettings;
    Result result = resultFromSl(
        slDLSSDGetOptimalSettings(options, nativeSettings),
        "slDLSSDGetOptimalSettings",
        log);
    if (!result) {
        return result;
    }

    settings = StreamlineDlssRrOptimalSettings{
        .renderWidth = nativeSettings.optimalRenderWidth,
        .renderHeight = nativeSettings.optimalRenderHeight,
        .renderWidthMin = nativeSettings.renderWidthMin,
        .renderHeightMin = nativeSettings.renderHeightMin,
        .renderWidthMax = nativeSettings.renderWidthMax,
        .renderHeightMax = nativeSettings.renderHeightMax,
    };
    const bool invalidRanges =
        (settings.renderWidthMin != 0 && settings.renderWidthMax != 0 &&
            settings.renderWidthMin > settings.renderWidthMax) ||
        (settings.renderHeightMin != 0 && settings.renderHeightMax != 0 &&
            settings.renderHeightMin > settings.renderHeightMax);
    const bool outsideReportedRange =
        (settings.renderWidthMin != 0 && settings.renderWidth < settings.renderWidthMin) ||
        (settings.renderHeightMin != 0 && settings.renderHeight < settings.renderHeightMin) ||
        (settings.renderWidthMax != 0 && settings.renderWidth > settings.renderWidthMax) ||
        (settings.renderHeightMax != 0 && settings.renderHeight > settings.renderHeightMax);
    if (settings.renderWidth == 0 ||
        settings.renderHeight == 0 ||
        settings.renderWidth > outputWidth ||
        settings.renderHeight > outputHeight ||
        invalidRanges ||
        outsideReportedRange) {
        log = "slDLSSDGetOptimalSettings returned invalid render dimensions " +
            std::to_string(settings.renderWidth) + "x" + std::to_string(settings.renderHeight) +
            " for output " + std::to_string(outputWidth) + "x" + std::to_string(outputHeight);
        settings = {};
        return makeError(Error::InvalidArgument);
    }

    state.dlssRrOptimalSettingsCache.push_back(DlssRrOptimalSettingsCacheEntry{
        .mode = mode,
        .outputWidth = outputWidth,
        .outputHeight = outputHeight,
        .settings = settings,
    });
    spdlog::info(
        "[Streamline] DLSS-RR optimal render extent {}x{} for output {}x{}",
        settings.renderWidth,
        settings.renderHeight,
        outputWidth,
        outputHeight);
    return {};
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

    const sl::Feature features[] = {
        sl::kFeatureDLSS,
        sl::kFeatureDLSS_RR,
    };

    sl::Preferences preferences;
    preferences.renderAPI = sl::RenderAPI::eVulkan;
    preferences.engine = sl::EngineType::eCustom;
    // Leave projectId/engineVersion empty so NGX uses the registered applicationId.
    preferences.applicationId = 231313132;
    // Leave pathsToPlugins unset so Streamline scans the executable directory,
    // where the build deploys only the DLSS-SR/DLSS-RR plugin set; Streamline
    // loads and signature-verifies every sl.*.dll it finds there before
    // filtering by requested features.
    preferences.featuresToLoad = features;
    preferences.numFeaturesToLoad = static_cast<uint32_t>(std::size(features));
    preferences.logLevel = sl::LogLevel::eDefault;
    preferences.logMessageCallback = streamlineLogCallback;
    preferences.flags =
        sl::PreferenceFlags::eDisableCLStateTracking |
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
    state.dlssSrOptimalSettingsCache.clear();
    state.dlssRrOptimalSettingsCache.clear();

    Result workaroundResult = initializeDescriptorHeapWorkaround(state, device, log);
    if (!workaroundResult) {
        state.vulkanDeviceSet = false;
        return workaroundResult;
    }

    sl::AdapterInfo adapterInfo;
    adapterInfo.vkPhysicalDevice = nativeHandleToVoid(device.physicalDevice);
    const sl::Result dlssSrSupportResult = slIsFeatureSupported(sl::kFeatureDLSS, adapterInfo);
    state.dlssSrSupported = dlssSrSupportResult == sl::Result::eOk;
    if (!state.dlssSrSupported) {
        log += "slIsFeatureSupported(kFeatureDLSS) returned ";
        log += slResultName(dlssSrSupportResult);
    }

    const sl::Result dlssRrSupportResult = slIsFeatureSupported(sl::kFeatureDLSS_RR, adapterInfo);
    state.dlssRrSupported = dlssRrSupportResult == sl::Result::eOk;
    if (!state.dlssRrSupported) {
        if (!log.empty() && log.back() != '\n') {
            log += '\n';
        }
        log += "slIsFeatureSupported(kFeatureDLSS_RR) returned ";
        log += slResultName(dlssRrSupportResult);
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

    if (state.vulkanDeviceSet) {
        if (state.dlssSrSupported) {
            (void)slFreeResources(sl::kFeatureDLSS, state.viewport);
        }
        if (state.dlssRrSupported) {
            (void)slFreeResources(sl::kFeatureDLSS_RR, state.viewport);
        }
    }
    (void)slShutdown();
    destroyDescriptorHeapWorkaround(state);
    state = StreamlineState{};
#endif
}

Result evaluateStreamlineDlssSr(CommandBuffer& commandBuffer, const StreamlineDlssSrDesc& desc, std::string& log)
{
#if !METALLIC_HAS_STREAMLINE
    (void)commandBuffer;
    (void)desc;
    log = "NVIDIA Streamline SDK is not available";
    return makeError(Error::Unsupported);
#else
    std::lock_guard lock(streamlineMutex());
    StreamlineState& state = streamlineState();
    if (!state.initialized || !state.vulkanDeviceSet || !state.dlssSrSupported) {
        log = "DLSS-SR is not available on the current Vulkan device";
        return makeError(Error::Unsupported);
    }
    if (desc.renderWidth == 0 ||
        desc.renderHeight == 0 ||
        desc.outputWidth == 0 ||
        desc.outputHeight == 0 ||
        desc.mode == StreamlineDlssSrMode::Off) {
        log = "DLSS-SR evaluate requires non-zero render/output dimensions and an enabled mode";
        return makeError(Error::InvalidArgument);
    }

    const auto optimalSettings = std::find_if(
        state.dlssSrOptimalSettingsCache.begin(),
        state.dlssSrOptimalSettingsCache.end(),
        [&desc](const DlssRrOptimalSettingsCacheEntry& entry) {
            return entry.mode == desc.mode &&
                entry.outputWidth == desc.outputWidth &&
                entry.outputHeight == desc.outputHeight;
        });
    if (optimalSettings == state.dlssSrOptimalSettingsCache.end() ||
        optimalSettings->settings.renderWidth != desc.renderWidth ||
        optimalSettings->settings.renderHeight != desc.renderHeight) {
        log = "DLSS-SR evaluate render dimensions do not match cached optimal settings";
        return makeError(Error::InvalidArgument);
    }

    const auto validateTexture = [&log](
                                     const StreamlineDlssSrTextureRef& ref,
                                     const char* name,
                                     uint32_t width,
                                     uint32_t height) {
        if (textureRefMatchesExtent(ref, width, height)) {
            return true;
        }
        log = std::string("DLSS-SR ") + name + " must be " +
            std::to_string(width) + "x" + std::to_string(height);
        return false;
    };
    if (!validateTexture(desc.inputColor, "inputColor", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.motionVectors, "motionVectors", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.depth, "depth", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.outputColor, "outputColor", desc.outputWidth, desc.outputHeight)) {
        return makeError(Error::InvalidArgument);
    }
    if (desc.motionVectors.texture->desc().format != Format::Rg16Sfloat) {
        log = "DLSS-SR motionVectors must use RG16_SFLOAT";
        return makeError(Error::InvalidArgument);
    }
    if (desc.depth.texture->desc().format != Format::D32Sfloat) {
        log = "DLSS-SR depth must use D32_SFLOAT";
        return makeError(Error::InvalidArgument);
    }
    if (desc.inputColor.texture->desc().format != Format::Rgba16Sfloat ||
        desc.outputColor.texture->desc().format != Format::Rgba16Sfloat) {
        log = "DLSS-SR inputColor and outputColor must use RGBA16_SFLOAT";
        return makeError(Error::InvalidArgument);
    }

    sl::CommandBuffer* nativeCommandBuffer = slCommandBuffer(commandBuffer);
    if (nativeCommandBuffer == nullptr) {
        log = "DLSS-SR evaluate received an invalid command buffer";
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

    sl::DLSSOptions options = makeDlssSrBaseOptions(desc.mode, desc.outputWidth, desc.outputHeight);
    result = resultFromSl(slDLSSSetOptions(state.viewport, options), "slDLSSSetOptions", log);
    if (!result) {
        return result;
    }

    sl::Constants constants = makeConstants(desc.camera, desc.reset);
    result = resultFromSl(slSetConstants(constants, *frameToken, state.viewport), "slSetConstants", log);
    if (!result) {
        return result;
    }

    sl::Extent renderExtent{
        .top = 0,
        .left = 0,
        .width = desc.renderWidth,
        .height = desc.renderHeight,
    };
    sl::Extent outputExtent{
        .top = 0,
        .left = 0,
        .width = desc.outputWidth,
        .height = desc.outputHeight,
    };
    std::vector<sl::Resource> resources;
    std::vector<sl::SubresourceRange> subresourceRanges;
    std::vector<sl::ResourceTag> tags;
    resources.reserve(4);
    subresourceRanges.reserve(4);
    tags.reserve(4);
    const bool validResources =
        appendResourceTag(desc.inputColor, sl::kBufferTypeScalingInputColor, renderExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.outputColor, sl::kBufferTypeScalingOutputColor, outputExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.motionVectors, sl::kBufferTypeMotionVectors, renderExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.depth, sl::kBufferTypeDepth, renderExtent, resources, subresourceRanges, tags);
    if (!validResources) {
        log = "DLSS-SR evaluate received invalid texture resources";
        return makeError(Error::InvalidArgument);
    }

    result = resultFromSl(
        slSetTagForFrame(*frameToken, state.viewport, tags.data(), static_cast<uint32_t>(tags.size()), nativeCommandBuffer),
        "slSetTagForFrame",
        log);
    if (!result) {
        return result;
    }

    prepareDescriptorStateForStreamline(state, commandBuffer);
    const sl::BaseStructure* inputs[] = {&state.viewport};
    return resultFromSl(
        slEvaluateFeature(
            sl::kFeatureDLSS,
            *frameToken,
            inputs,
            static_cast<uint32_t>(std::size(inputs)),
            nativeCommandBuffer),
        "slEvaluateFeature(kFeatureDLSS)",
        log);
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
    if (desc.renderWidth == 0 ||
        desc.renderHeight == 0 ||
        desc.outputWidth == 0 ||
        desc.outputHeight == 0 ||
        desc.mode == StreamlineDlssRrMode::Off) {
        log = "DLSS-RR evaluate requires non-zero render/output dimensions and an enabled mode";
        return makeError(Error::InvalidArgument);
    }

    const auto optimalSettings = std::find_if(
        state.dlssRrOptimalSettingsCache.begin(),
        state.dlssRrOptimalSettingsCache.end(),
        [&desc](const DlssRrOptimalSettingsCacheEntry& entry) {
            return entry.mode == desc.mode &&
                entry.outputWidth == desc.outputWidth &&
                entry.outputHeight == desc.outputHeight;
        });
    if (optimalSettings == state.dlssRrOptimalSettingsCache.end() ||
        optimalSettings->settings.renderWidth != desc.renderWidth ||
        optimalSettings->settings.renderHeight != desc.renderHeight) {
        log = "DLSS-RR evaluate render dimensions do not match cached optimal settings";
        return makeError(Error::InvalidArgument);
    }

    const auto validateTexture = [&log](
                                     const StreamlineDlssRrTextureRef& ref,
                                     const char* name,
                                     uint32_t width,
                                     uint32_t height) {
        if (textureRefMatchesExtent(ref, width, height)) {
            return true;
        }
        log = std::string("DLSS-RR ") + name + " must be " +
            std::to_string(width) + "x" + std::to_string(height);
        return false;
    };
    if (!validateTexture(desc.inputColor, "inputColor", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.albedo, "albedo", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.specularAlbedo, "specularAlbedo", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.normalRoughness, "normalRoughness", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.motionVectors, "motionVectors", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.linearDepth, "linearDepth", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.specularHitDistance, "specularHitDistance", desc.renderWidth, desc.renderHeight) ||
        !validateTexture(desc.outputColor, "outputColor", desc.outputWidth, desc.outputHeight)) {
        return makeError(Error::InvalidArgument);
    }
    if (desc.motionVectors.texture->desc().format != Format::Rg16Sfloat) {
        log = "DLSS-RR motionVectors must use RG16_SFLOAT";
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

    sl::Constants constants = makeConstants(desc.camera, desc.reset);
    result = resultFromSl(slSetConstants(constants, *frameToken, state.viewport), "slSetConstants", log);
    if (!result) {
        return result;
    }

    sl::Extent renderExtent{
        .top = 0,
        .left = 0,
        .width = desc.renderWidth,
        .height = desc.renderHeight,
    };
    sl::Extent outputExtent{
        .top = 0,
        .left = 0,
        .width = desc.outputWidth,
        .height = desc.outputHeight,
    };
    std::vector<sl::Resource> resources;
    std::vector<sl::SubresourceRange> subresourceRanges;
    std::vector<sl::ResourceTag> tags;
    resources.reserve(8);
    subresourceRanges.reserve(8);
    tags.reserve(8);
    const bool validResources =
        appendResourceTag(desc.inputColor, sl::kBufferTypeScalingInputColor, renderExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.outputColor, sl::kBufferTypeScalingOutputColor, outputExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.albedo, sl::kBufferTypeAlbedo, renderExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.specularAlbedo, sl::kBufferTypeSpecularAlbedo, renderExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.normalRoughness, sl::kBufferTypeNormalRoughness, renderExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.motionVectors, sl::kBufferTypeMotionVectors, renderExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.linearDepth, sl::kBufferTypeLinearDepth, renderExtent, resources, subresourceRanges, tags) &&
        appendResourceTag(desc.specularHitDistance, sl::kBufferTypeSpecularHitDistance, renderExtent, resources, subresourceRanges, tags);
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

    prepareDescriptorStateForStreamline(state, commandBuffer);
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
