#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/GAPI/PipelineCacheFile.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/Profiling/NsightAftermath.h"
#include "Runtime/Render/Profiling/NsightEvents.h"
#include "Runtime/Render/SlangCompiler.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_loadso.h>
#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_VULKAN_VERSION 1004000
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

constexpr uint32_t kVulkanApiVersion = VK_API_VERSION_1_4;
constexpr uint64_t kAcquireTimeoutNanoseconds = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kVulkanPipelineCacheBackendTag = 0x4b56544du;

struct StateInfo {
    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

Result resultFromVk(VkResult result)
{
    switch (result) {
    case VK_SUCCESS:
        return {};
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return makeError(Error::OutOfMemory);
    case VK_ERROR_DEVICE_LOST:
        profiling::handleNsightAftermathDeviceLost();
        return makeError(Error::DeviceLost);
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
        return makeError(Error::OutOfDate);
    case VK_ERROR_EXTENSION_NOT_PRESENT:
    case VK_ERROR_FEATURE_NOT_PRESENT:
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
    case VK_ERROR_INCOMPATIBLE_DRIVER:
    case VK_ERROR_LAYER_NOT_PRESENT:
        return makeError(Error::Unsupported);
    default:
        return makeError(Error::Failure);
    }
}

bool hasName(const std::vector<VkExtensionProperties>& properties, const char* name)
{
    return std::any_of(properties.begin(), properties.end(), [name](const VkExtensionProperties& property) {
        return std::strcmp(property.extensionName, name) == 0;
    });
}

bool hasName(const std::vector<VkLayerProperties>& properties, const char* name)
{
    return std::any_of(properties.begin(), properties.end(), [name](const VkLayerProperties& property) {
        return std::strcmp(property.layerName, name) == 0;
    });
}

std::mutex& sdlVulkanLibraryMutex()
{
    static std::mutex mutex;
    return mutex;
}

uint32_t& sdlVulkanLibraryRefCount()
{
    static uint32_t refCount = 0;
    return refCount;
}

bool acquireSdlVulkanLibrary(const char* libraryPath = nullptr)
{
    std::lock_guard lock(sdlVulkanLibraryMutex());
    uint32_t& refCount = sdlVulkanLibraryRefCount();
    if (refCount == 0 && !SDL_Vulkan_LoadLibrary(libraryPath)) {
        return false;
    }

    ++refCount;
    return true;
}

void releaseSdlVulkanLibrary()
{
    std::lock_guard lock(sdlVulkanLibraryMutex());
    uint32_t& refCount = sdlVulkanLibraryRefCount();
    if (refCount == 0) {
        return;
    }

    --refCount;
    if (refCount == 0) {
        SDL_Vulkan_UnloadLibrary();
    }
}

PFN_vkGetInstanceProcAddr loadVulkanLoaderProcAddr(
    const char* libraryName,
    SDL_SharedObject*& outLibraryHandle)
{
    outLibraryHandle = nullptr;

    SDL_SharedObject* const libraryHandle = SDL_LoadObject(libraryName);
    if (libraryHandle == nullptr) {
        return nullptr;
    }

    SDL_FunctionPointer const procAddr = SDL_LoadFunction(libraryHandle, "vkGetInstanceProcAddr");
    if (procAddr == nullptr) {
        SDL_UnloadObject(libraryHandle);
        return nullptr;
    }

    outLibraryHandle = libraryHandle;
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(procAddr);
}

std::mutex& volkDeviceMutex()
{
    static std::mutex mutex;
    return mutex;
}

VkDevice& activeVolkDevice()
{
    static VkDevice device = VK_NULL_HANDLE;
    return device;
}

void activateVolkDevice(VkDevice device)
{
    if (device == VK_NULL_HANDLE) {
        return;
    }

    std::lock_guard lock(volkDeviceMutex());
    VkDevice& activeDevice = activeVolkDevice();
    if (activeDevice != device) {
        volkLoadDevice(device);
        activeDevice = device;
    }
}

void clearActiveVolkDevice(VkDevice device)
{
    std::lock_guard lock(volkDeviceMutex());
    VkDevice& activeDevice = activeVolkDevice();
    if (activeDevice == device) {
        activeDevice = VK_NULL_HANDLE;
    }
}

std::vector<VkExtensionProperties> enumerateInstanceExtensions()
{
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    if (count > 0) {
        vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
    }
    return extensions;
}

std::vector<VkLayerProperties> enumerateInstanceLayers()
{
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    if (count > 0) {
        vkEnumerateInstanceLayerProperties(&count, layers.data());
    }
    return layers;
}

std::vector<VkExtensionProperties> enumerateDeviceExtensions(VkPhysicalDevice physicalDevice)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    if (count > 0) {
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data());
    }
    return extensions;
}

VkFormat toVkFormat(Format format)
{
    switch (format) {
    case Format::R8Unorm:
        return VK_FORMAT_R8_UNORM;
    case Format::R8Snorm:
        return VK_FORMAT_R8_SNORM;
    case Format::R8Uint:
        return VK_FORMAT_R8_UINT;
    case Format::R8Sint:
        return VK_FORMAT_R8_SINT;
    case Format::Rg8Unorm:
        return VK_FORMAT_R8G8_UNORM;
    case Format::Rg8Snorm:
        return VK_FORMAT_R8G8_SNORM;
    case Format::Rg8Uint:
        return VK_FORMAT_R8G8_UINT;
    case Format::Rg8Sint:
        return VK_FORMAT_R8G8_SINT;
    // LibNTC stores latent features as packed 4-bit BGRA channels.
    case Format::Bgra4Unorm:
        return VK_FORMAT_A4R4G4B4_UNORM_PACK16;
    case Format::Bgra8Unorm:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case Format::Bgra8Srgb:
        return VK_FORMAT_B8G8R8A8_SRGB;
    case Format::Rgba8Unorm:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::Rgba8Snorm:
        return VK_FORMAT_R8G8B8A8_SNORM;
    case Format::Rgba8Srgb:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case Format::Rgba8Uint:
        return VK_FORMAT_R8G8B8A8_UINT;
    case Format::Rgba8Sint:
        return VK_FORMAT_R8G8B8A8_SINT;
    case Format::R16Unorm:
        return VK_FORMAT_R16_UNORM;
    case Format::R16Snorm:
        return VK_FORMAT_R16_SNORM;
    case Format::R16Uint:
        return VK_FORMAT_R16_UINT;
    case Format::R16Sint:
        return VK_FORMAT_R16_SINT;
    case Format::R16Sfloat:
        return VK_FORMAT_R16_SFLOAT;
    case Format::Rg16Unorm:
        return VK_FORMAT_R16G16_UNORM;
    case Format::Rg16Snorm:
        return VK_FORMAT_R16G16_SNORM;
    case Format::Rg16Uint:
        return VK_FORMAT_R16G16_UINT;
    case Format::Rg16Sint:
        return VK_FORMAT_R16G16_SINT;
    case Format::Rg16Sfloat:
        return VK_FORMAT_R16G16_SFLOAT;
    case Format::Rgba16Unorm:
        return VK_FORMAT_R16G16B16A16_UNORM;
    case Format::Rgba16Snorm:
        return VK_FORMAT_R16G16B16A16_SNORM;
    case Format::Rgba16Uint:
        return VK_FORMAT_R16G16B16A16_UINT;
    case Format::Rgba16Sint:
        return VK_FORMAT_R16G16B16A16_SINT;
    case Format::Rgba16Sfloat:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Format::R32Uint:
        return VK_FORMAT_R32_UINT;
    case Format::R32Sint:
        return VK_FORMAT_R32_SINT;
    case Format::R32Sfloat:
        return VK_FORMAT_R32_SFLOAT;
    case Format::Rg32Uint:
        return VK_FORMAT_R32G32_UINT;
    case Format::Rg32Sint:
        return VK_FORMAT_R32G32_SINT;
    case Format::Rg32Sfloat:
        return VK_FORMAT_R32G32_SFLOAT;
    case Format::Rgb32Uint:
        return VK_FORMAT_R32G32B32_UINT;
    case Format::Rgb32Sint:
        return VK_FORMAT_R32G32B32_SINT;
    case Format::Rgb32Sfloat:
        return VK_FORMAT_R32G32B32_SFLOAT;
    case Format::Rgba32Uint:
        return VK_FORMAT_R32G32B32A32_UINT;
    case Format::Rgba32Sint:
        return VK_FORMAT_R32G32B32A32_SINT;
    case Format::Rgba32Sfloat:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case Format::A2B10G10R10UnormPack32:
        return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case Format::A2R10G10B10UintPack32:
        return VK_FORMAT_A2R10G10B10_UINT_PACK32;
    case Format::B10G11R11UfloatPack32:
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case Format::E5B9G9R9UfloatPack32:
        return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case Format::D32Sfloat:
        return VK_FORMAT_D32_SFLOAT;
    case Format::Unknown:
        return VK_FORMAT_UNDEFINED;
    }

    return VK_FORMAT_UNDEFINED;
}

Format fromVkFormat(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R8_UNORM:
        return Format::R8Unorm;
    case VK_FORMAT_R8_SNORM:
        return Format::R8Snorm;
    case VK_FORMAT_R8_UINT:
        return Format::R8Uint;
    case VK_FORMAT_R8_SINT:
        return Format::R8Sint;
    case VK_FORMAT_R8G8_UNORM:
        return Format::Rg8Unorm;
    case VK_FORMAT_R8G8_SNORM:
        return Format::Rg8Snorm;
    case VK_FORMAT_R8G8_UINT:
        return Format::Rg8Uint;
    case VK_FORMAT_R8G8_SINT:
        return Format::Rg8Sint;
    case VK_FORMAT_A4R4G4B4_UNORM_PACK16:
        return Format::Bgra4Unorm;
    case VK_FORMAT_B8G8R8A8_UNORM:
        return Format::Bgra8Unorm;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return Format::Bgra8Srgb;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return Format::Rgba8Unorm;
    case VK_FORMAT_R8G8B8A8_SNORM:
        return Format::Rgba8Snorm;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return Format::Rgba8Srgb;
    case VK_FORMAT_R8G8B8A8_UINT:
        return Format::Rgba8Uint;
    case VK_FORMAT_R8G8B8A8_SINT:
        return Format::Rgba8Sint;
    case VK_FORMAT_R16_UNORM:
        return Format::R16Unorm;
    case VK_FORMAT_R16_SNORM:
        return Format::R16Snorm;
    case VK_FORMAT_R16_UINT:
        return Format::R16Uint;
    case VK_FORMAT_R16_SINT:
        return Format::R16Sint;
    case VK_FORMAT_R16_SFLOAT:
        return Format::R16Sfloat;
    case VK_FORMAT_R16G16_UNORM:
        return Format::Rg16Unorm;
    case VK_FORMAT_R16G16_SNORM:
        return Format::Rg16Snorm;
    case VK_FORMAT_R16G16_UINT:
        return Format::Rg16Uint;
    case VK_FORMAT_R16G16_SINT:
        return Format::Rg16Sint;
    case VK_FORMAT_R16G16_SFLOAT:
        return Format::Rg16Sfloat;
    case VK_FORMAT_R16G16B16A16_UNORM:
        return Format::Rgba16Unorm;
    case VK_FORMAT_R16G16B16A16_SNORM:
        return Format::Rgba16Snorm;
    case VK_FORMAT_R16G16B16A16_UINT:
        return Format::Rgba16Uint;
    case VK_FORMAT_R16G16B16A16_SINT:
        return Format::Rgba16Sint;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return Format::Rgba16Sfloat;
    case VK_FORMAT_R32_UINT:
        return Format::R32Uint;
    case VK_FORMAT_R32_SINT:
        return Format::R32Sint;
    case VK_FORMAT_R32_SFLOAT:
        return Format::R32Sfloat;
    case VK_FORMAT_R32G32_UINT:
        return Format::Rg32Uint;
    case VK_FORMAT_R32G32_SINT:
        return Format::Rg32Sint;
    case VK_FORMAT_R32G32_SFLOAT:
        return Format::Rg32Sfloat;
    case VK_FORMAT_R32G32B32_UINT:
        return Format::Rgb32Uint;
    case VK_FORMAT_R32G32B32_SINT:
        return Format::Rgb32Sint;
    case VK_FORMAT_R32G32B32_SFLOAT:
        return Format::Rgb32Sfloat;
    case VK_FORMAT_R32G32B32A32_UINT:
        return Format::Rgba32Uint;
    case VK_FORMAT_R32G32B32A32_SINT:
        return Format::Rgba32Sint;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return Format::Rgba32Sfloat;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return Format::A2B10G10R10UnormPack32;
    case VK_FORMAT_A2R10G10B10_UINT_PACK32:
        return Format::A2R10G10B10UintPack32;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return Format::B10G11R11UfloatPack32;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        return Format::E5B9G9R9UfloatPack32;
    case VK_FORMAT_D32_SFLOAT:
        return Format::D32Sfloat;
    default:
        return Format::Unknown;
    }
}

VkImageAspectFlags aspectForFormat(Format format)
{
    if (format == Format::D32Sfloat) {
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

VkBufferUsageFlags toVkBufferUsage(BufferUsageBits usage)
{
    VkBufferUsageFlags flags = 0;
    if (hasFlag(usage, BufferUsageBits::Vertex)) {
        flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }
    if (hasFlag(usage, BufferUsageBits::Index)) {
        flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }
    if (hasFlag(usage, BufferUsageBits::Constant)) {
        flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }
    if (hasFlag(usage, BufferUsageBits::Storage)) {
        flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if (hasFlag(usage, BufferUsageBits::TransferSource)) {
        flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    if (hasFlag(usage, BufferUsageBits::TransferDestination)) {
        flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if (hasFlag(usage, BufferUsageBits::ShaderDeviceAddress)) {
        flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    if (hasFlag(usage, BufferUsageBits::AccelerationStructureBuildInput)) {
        flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }
    if (hasFlag(usage, BufferUsageBits::AccelerationStructureStorage)) {
        flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    }
    if (hasFlag(usage, BufferUsageBits::Indirect)) {
        flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    }
    return flags != 0 ? flags : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
}

VkImageUsageFlags toVkImageUsage(TextureUsageBits usage)
{
    VkImageUsageFlags flags = 0;
    if (hasFlag(usage, TextureUsageBits::Sampled)) {
        flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (hasFlag(usage, TextureUsageBits::Storage)) {
        flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (hasFlag(usage, TextureUsageBits::ColorAttachment)) {
        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if (hasFlag(usage, TextureUsageBits::DepthStencilAttachment)) {
        flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    if (hasFlag(usage, TextureUsageBits::TransferSource)) {
        flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    if (hasFlag(usage, TextureUsageBits::TransferDestination)) {
        flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }
    return flags != 0 ? flags : VK_IMAGE_USAGE_SAMPLED_BIT;
}

VkFilter toVkSamplerFilter(SamplerFilter filter)
{
    return filter == SamplerFilter::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
}

VkSamplerMipmapMode toVkSamplerMipmapMode(SamplerFilter filter)
{
    return filter == SamplerFilter::Nearest
        ? VK_SAMPLER_MIPMAP_MODE_NEAREST
        : VK_SAMPLER_MIPMAP_MODE_LINEAR;
}

VkSamplerAddressMode toVkSamplerAddressMode(SamplerAddressMode mode)
{
    switch (mode) {
    case SamplerAddressMode::Repeat:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case SamplerAddressMode::MirroredRepeat:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case SamplerAddressMode::ClampToEdge:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case SamplerAddressMode::ClampToBorder:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    }
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
}

VkAccelerationStructureTypeKHR toVkAccelerationStructureType(
    RayTracingAccelerationStructureType type)
{
    return type == RayTracingAccelerationStructureType::TopLevel
        ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
        : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
}

VkBuildAccelerationStructureFlagsKHR toVkAccelerationStructureBuildFlags(
    RayTracingAccelerationStructureBuildFlags flags)
{
    VkBuildAccelerationStructureFlagsKHR result = 0;
    if (hasFlag(flags, RayTracingAccelerationStructureBuildFlags::PreferFastTrace)) {
        result |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    }
    if (hasFlag(flags, RayTracingAccelerationStructureBuildFlags::PreferFastBuild)) {
        result |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    }
    if (hasFlag(flags, RayTracingAccelerationStructureBuildFlags::AllowUpdate)) {
        result |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    if (hasFlag(flags, RayTracingAccelerationStructureBuildFlags::AllowCompaction)) {
        result |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    }
    return result;
}

VkGeometryFlagsKHR toVkGeometryFlags(RayTracingGeometryFlags flags)
{
    VkGeometryFlagsKHR result = 0;
    if (hasFlag(flags, RayTracingGeometryFlags::Opaque)) {
        result |= VK_GEOMETRY_OPAQUE_BIT_KHR;
    }
    if (hasFlag(flags, RayTracingGeometryFlags::NoDuplicateAnyHitInvocation)) {
        result |= VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    }
    return result;
}

VkGeometryInstanceFlagsKHR toVkInstanceFlags(RayTracingInstanceFlags flags)
{
    VkGeometryInstanceFlagsKHR result = 0;
    if (hasFlag(flags, RayTracingInstanceFlags::TriangleFacingCullDisable)) {
        result |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }
    if (hasFlag(flags, RayTracingInstanceFlags::TriangleFrontCounterClockwise)) {
        result |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
    }
    if (hasFlag(flags, RayTracingInstanceFlags::ForceOpaque)) {
        result |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    }
    if (hasFlag(flags, RayTracingInstanceFlags::ForceNonOpaque)) {
        result |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
    }
    return result;
}

VkIndexType toVkRayTracingIndexType(RayTracingIndexType type)
{
    switch (type) {
    case RayTracingIndexType::None:
        return VK_INDEX_TYPE_NONE_KHR;
    case RayTracingIndexType::Uint16:
        return VK_INDEX_TYPE_UINT16;
    case RayTracingIndexType::Uint32:
        return VK_INDEX_TYPE_UINT32;
    }
    return VK_INDEX_TYPE_NONE_KHR;
}

#ifdef VK_NV_cluster_acceleration_structure
VkClusterAccelerationStructureIndexFormatFlagBitsNV toVkClusterIndexFormat(
    ClusterAccelerationStructureIndexFormat format)
{
    switch (format) {
    case ClusterAccelerationStructureIndexFormat::Uint8:
        return VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
    case ClusterAccelerationStructureIndexFormat::Uint16:
        return VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV;
    case ClusterAccelerationStructureIndexFormat::Uint32:
        return VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV;
    }
    return VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
}

uint64_t clusterIndexByteSize(ClusterAccelerationStructureIndexFormat format)
{
    switch (format) {
    case ClusterAccelerationStructureIndexFormat::Uint8:
        return 1;
    case ClusterAccelerationStructureIndexFormat::Uint16:
        return 2;
    case ClusterAccelerationStructureIndexFormat::Uint32:
        return 4;
    }
    return 0;
}
#endif

uint32_t formatTexelByteSize(Format format)
{
    switch (format) {
    case Format::R8Unorm:
    case Format::R8Snorm:
    case Format::R8Uint:
    case Format::R8Sint:
        return 1;
    case Format::Rg8Unorm:
    case Format::Rg8Snorm:
    case Format::Rg8Uint:
    case Format::Rg8Sint:
    case Format::Bgra4Unorm:
    case Format::R16Unorm:
    case Format::R16Snorm:
    case Format::R16Uint:
    case Format::R16Sint:
    case Format::R16Sfloat:
        return 2;
    case Format::Bgra8Unorm:
    case Format::Bgra8Srgb:
    case Format::Rgba8Unorm:
    case Format::Rgba8Snorm:
    case Format::Rgba8Srgb:
    case Format::Rgba8Uint:
    case Format::Rgba8Sint:
    case Format::Rg16Unorm:
    case Format::Rg16Snorm:
    case Format::Rg16Uint:
    case Format::Rg16Sint:
    case Format::Rg16Sfloat:
    case Format::R32Uint:
    case Format::R32Sint:
    case Format::R32Sfloat:
    case Format::A2B10G10R10UnormPack32:
    case Format::A2R10G10B10UintPack32:
    case Format::B10G11R11UfloatPack32:
    case Format::E5B9G9R9UfloatPack32:
    case Format::D32Sfloat:
        return 4;
    case Format::Rgba16Unorm:
    case Format::Rgba16Snorm:
    case Format::Rgba16Uint:
    case Format::Rgba16Sint:
    case Format::Rgba16Sfloat:
    case Format::Rg32Uint:
    case Format::Rg32Sint:
    case Format::Rg32Sfloat:
        return 8;
    case Format::Rgb32Uint:
    case Format::Rgb32Sint:
    case Format::Rgb32Sfloat:
        return 12;
    case Format::Rgba32Uint:
    case Format::Rgba32Sint:
    case Format::Rgba32Sfloat:
        return 16;
    case Format::Unknown:
        break;
    }
    return 0;
}

bool fillBufferImageLayout(
    Format format,
    uint32_t width,
    uint32_t height,
    uint32_t bufferRowPitch,
    uint32_t bufferSlicePitch,
    uint32_t& outBufferRowLength,
    uint32_t& outBufferImageHeight)
{
    outBufferRowLength = 0;
    outBufferImageHeight = 0;
    if (bufferRowPitch == 0 && bufferSlicePitch == 0) {
        return true;
    }

    const uint32_t bytesPerTexel = formatTexelByteSize(format);
    if (bytesPerTexel == 0) {
        return false;
    }

    const uint64_t tightRowPitch = static_cast<uint64_t>(width) * bytesPerTexel;
    const uint64_t rowPitch = bufferRowPitch == 0
        ? tightRowPitch
        : static_cast<uint64_t>(bufferRowPitch);
    if (rowPitch < tightRowPitch || rowPitch % bytesPerTexel != 0) {
        return false;
    }
    outBufferRowLength = bufferRowPitch == 0
        ? 0
        : static_cast<uint32_t>(rowPitch / bytesPerTexel);

    if (bufferSlicePitch != 0) {
        const uint64_t tightSlicePitch = rowPitch * height;
        if (bufferSlicePitch < tightSlicePitch || bufferSlicePitch % rowPitch != 0) {
            return false;
        }
        outBufferImageHeight = static_cast<uint32_t>(bufferSlicePitch / rowPitch);
    }
    return true;
}

VkImageType toVkImageType(TextureType type)
{
    switch (type) {
    case TextureType::Texture1D:
        return VK_IMAGE_TYPE_1D;
    case TextureType::Texture2D:
        return VK_IMAGE_TYPE_2D;
    case TextureType::Texture3D:
        return VK_IMAGE_TYPE_3D;
    }

    return VK_IMAGE_TYPE_2D;
}

VkImageViewType toVkImageViewType(TextureType type)
{
    switch (type) {
    case TextureType::Texture1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case TextureType::Texture2D:
        return VK_IMAGE_VIEW_TYPE_2D;
    case TextureType::Texture3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    }

    return VK_IMAGE_VIEW_TYPE_2D;
}

VkAttachmentLoadOp toVkLoadOp(LoadOp loadOp)
{
    switch (loadOp) {
    case LoadOp::Load:
        return VK_ATTACHMENT_LOAD_OP_LOAD;
    case LoadOp::Clear:
        return VK_ATTACHMENT_LOAD_OP_CLEAR;
    case LoadOp::DontCare:
        return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }

    return VK_ATTACHMENT_LOAD_OP_LOAD;
}

VkAttachmentStoreOp toVkStoreOp(StoreOp storeOp)
{
    switch (storeOp) {
    case StoreOp::Store:
        return VK_ATTACHMENT_STORE_OP_STORE;
    case StoreOp::DontCare:
        return VK_ATTACHMENT_STORE_OP_DONT_CARE;
    }

    return VK_ATTACHMENT_STORE_OP_STORE;
}

VkPrimitiveTopology toVkPrimitiveTopology(PrimitiveTopology topology)
{
    switch (topology) {
    case PrimitiveTopology::TriangleList:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }

    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

VkCullModeFlags toVkCullMode(CullMode cullMode)
{
    switch (cullMode) {
    case CullMode::None:
        return VK_CULL_MODE_NONE;
    case CullMode::Front:
        return VK_CULL_MODE_FRONT_BIT;
    case CullMode::Back:
        return VK_CULL_MODE_BACK_BIT;
    }

    return VK_CULL_MODE_NONE;
}

VkFrontFace toVkFrontFace(FrontFace frontFace)
{
    switch (frontFace) {
    case FrontFace::CounterClockwise:
        return VK_FRONT_FACE_COUNTER_CLOCKWISE;
    case FrontFace::Clockwise:
        return VK_FRONT_FACE_CLOCKWISE;
    }

    return VK_FRONT_FACE_COUNTER_CLOCKWISE;
}

VkCompareOp toVkCompareOp(CompareOp compareOp)
{
    switch (compareOp) {
    case CompareOp::Never:
        return VK_COMPARE_OP_NEVER;
    case CompareOp::Less:
        return VK_COMPARE_OP_LESS;
    case CompareOp::Equal:
        return VK_COMPARE_OP_EQUAL;
    case CompareOp::LessEqual:
        return VK_COMPARE_OP_LESS_OR_EQUAL;
    case CompareOp::Greater:
        return VK_COMPARE_OP_GREATER;
    case CompareOp::NotEqual:
        return VK_COMPARE_OP_NOT_EQUAL;
    case CompareOp::GreaterEqual:
        return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case CompareOp::Always:
        return VK_COMPARE_OP_ALWAYS;
    }

    return VK_COMPARE_OP_LESS_OR_EQUAL;
}

StateInfo stateInfo(ResourceState state)
{
    const VkPipelineStageFlags2 shaderReadStages =
        VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    switch (state) {
    case ResourceState::Undefined:
        return {
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_IMAGE_LAYOUT_UNDEFINED,
        };
    case ResourceState::Present:
        return {
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
    case ResourceState::ColorAttachment:
        return {
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
    case ResourceState::DepthStencilAttachment:
        return {
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
    case ResourceState::ShaderRead:
        return {
            shaderReadStages,
            VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
    case ResourceState::IndirectArgument:
        return {
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
        };
    case ResourceState::TransferSource:
        return {
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        };
    case ResourceState::TransferDestination:
        return {
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        };
    case ResourceState::General:
        return {
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    return {};
}

VkPipelineStageFlags2 toVkPipelineStages(PipelineStageBits stages)
{
    VkPipelineStageFlags2 flags = VK_PIPELINE_STAGE_2_NONE;
    const auto value = static_cast<uint64_t>(stages);
    if ((value & static_cast<uint64_t>(PipelineStageBits::TopOfPipe)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::DrawIndirect)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::VertexShader)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::FragmentShader)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::ComputeShader)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::ColorAttachment)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::Transfer)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::BottomOfPipe)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::AllCommands)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
    return flags != VK_PIPELINE_STAGE_2_NONE ? flags : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
}

VmaAllocationCreateInfo allocationInfoForMemory(MemoryLocation location)
{
    VmaAllocationCreateInfo info{};
    info.usage = VMA_MEMORY_USAGE_AUTO;

    switch (location) {
    case MemoryLocation::Device:
        info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case MemoryLocation::HostUpload:
        info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        break;
    case MemoryLocation::HostReadback:
        info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        break;
    }

    return info;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*)
{
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0 ||
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        spdlog::warn("Vulkan validation: {}", callbackData->pMessage);
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance)
{
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (create == nullptr) {
        return VK_NULL_HANDLE;
    }

    VkDebugUtilsMessengerCreateInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (create(instance, &info, nullptr, &messenger) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    return messenger;
}

void destroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger)
{
    if (messenger == VK_NULL_HANDLE) {
        return;
    }

    auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy != nullptr) {
        destroy(instance, messenger, nullptr);
    }
}

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t capacityFromBytes(VkDeviceSize byteSize, VkDeviceSize descriptorSize)
{
    if (descriptorSize == 0) {
        return 0;
    }

    constexpr VkDeviceSize kMaxUint32 = std::numeric_limits<uint32_t>::max();
    return static_cast<uint32_t>(std::min(byteSize / descriptorSize, kMaxUint32));
}

struct BindlessHeapPushConstants {
    uint32_t imageShaderIndexBase = 0;
    uint32_t bufferShaderIndexBase = 0;
};

static_assert(sizeof(BindlessHeapPushConstants) == 8);

class DescriptorHeapWriter {
public:
    static bool isSupported(VkPhysicalDevice physicalDevice)
    {
        VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
        };
        VkPhysicalDeviceFeatures2 features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &heapFeatures,
        };
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
        if (heapFeatures.descriptorHeap != VK_TRUE) {
            return false;
        }

        return hasUsableProperties(physicalDevice);
    }

    static bool hasUsableProperties(VkPhysicalDevice physicalDevice)
    {
        VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProperties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
        };
        VkPhysicalDeviceProperties2 properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &heapProperties,
        };
        vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
        return heapProperties.samplerDescriptorSize > 0 &&
            heapProperties.imageDescriptorSize > 0 &&
            heapProperties.bufferDescriptorSize > 0 &&
            heapProperties.maxPushDataSize >= sizeof(BindlessHeapPushConstants);
    }

    VkResult initialize(VkPhysicalDevice physicalDevice, VkDevice device)
    {
        *this = {};
        if (device == VK_NULL_HANDLE) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProperties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
        };
        VkPhysicalDeviceProperties2 properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &heapProperties,
        };
        vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
        if (heapProperties.samplerDescriptorSize == 0 ||
            heapProperties.imageDescriptorSize == 0 ||
            heapProperties.bufferDescriptorSize == 0 ||
            heapProperties.maxPushDataSize < sizeof(BindlessHeapPushConstants)) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        device_ = device;
        samplerDescriptorSize_ = heapProperties.samplerDescriptorSize;
        imageDescriptorSize_ = heapProperties.imageDescriptorSize;
        bufferDescriptorSize_ = heapProperties.bufferDescriptorSize;
        samplerDescriptorAlignment_ = heapProperties.samplerDescriptorAlignment;
        imageDescriptorAlignment_ = heapProperties.imageDescriptorAlignment;
        bufferDescriptorAlignment_ = heapProperties.bufferDescriptorAlignment;
        samplerHeapAlignment_ = heapProperties.samplerHeapAlignment;
        resourceHeapAlignment_ = heapProperties.resourceHeapAlignment;
        maxSamplerHeapSize_ = heapProperties.maxSamplerHeapSize;
        maxResourceHeapSize_ = heapProperties.maxResourceHeapSize;
        minSamplerHeapReservedRange_ = heapProperties.minSamplerHeapReservedRange;
        minResourceHeapReservedRange_ = heapProperties.minResourceHeapReservedRange;
        maxPushDataSize_ = heapProperties.maxPushDataSize;
        return VK_SUCCESS;
    }

    bool initialized() const { return device_ != VK_NULL_HANDLE; }

    VkDeviceSize samplerDescriptorSize() const { return samplerDescriptorSize_; }
    VkDeviceSize imageDescriptorSize() const { return imageDescriptorSize_; }
    VkDeviceSize bufferDescriptorSize() const { return bufferDescriptorSize_; }
    VkDeviceSize samplerHeapAlignment() const { return samplerHeapAlignment_; }
    VkDeviceSize resourceHeapAlignment() const { return resourceHeapAlignment_; }
    VkDeviceSize maxSamplerHeapSize() const { return maxSamplerHeapSize_; }
    VkDeviceSize maxResourceHeapSize() const { return maxResourceHeapSize_; }
    VkDeviceSize minSamplerHeapReservedRange() const { return minSamplerHeapReservedRange_; }
    VkDeviceSize minResourceHeapReservedRange() const { return minResourceHeapReservedRange_; }
    VkDeviceSize maxPushDataSize() const { return maxPushDataSize_; }

    VkDeviceSize samplerOffset(uint32_t index) const { return samplerDescriptorSize_ * index; }
    VkDeviceSize imageOffset(uint32_t index) const { return imageDescriptorSize_ * index; }
    VkDeviceSize bufferOffset(uint32_t index) const { return bufferDescriptorSize_ * index; }

    VkDeviceSize appendSamplerDescriptors(VkDeviceSize& offset, uint32_t count) const
    {
        const VkDeviceSize start = alignUp(offset, samplerDescriptorAlignment_);
        offset = start + samplerDescriptorSize_ * count;
        return start;
    }

    VkDeviceSize appendImageDescriptors(VkDeviceSize& offset, uint32_t count) const
    {
        const VkDeviceSize start = alignUp(offset, imageDescriptorAlignment_);
        offset = start + imageDescriptorSize_ * count;
        return start;
    }

    VkDeviceSize appendBufferDescriptors(VkDeviceSize& offset, uint32_t count) const
    {
        const VkDeviceSize start = alignUp(offset, bufferDescriptorAlignment_);
        offset = start + bufferDescriptorSize_ * count;
        return start;
    }

    VkDeviceSize appendSamplerReservedRange(VkDeviceSize& offset) const
    {
        const VkDeviceSize start = offset;
        offset = start + minSamplerHeapReservedRange_;
        return start;
    }

    VkDeviceSize appendResourceReservedRange(VkDeviceSize& offset) const
    {
        const VkDeviceSize start = alignUp(offset, imageDescriptorAlignment_);
        offset = start + minResourceHeapReservedRange_;
        return start;
    }

    VkDeviceSize alignToSamplerHeap(VkDeviceSize offset) const { return alignUp(offset, samplerHeapAlignment_); }
    VkDeviceSize alignToResourceHeap(VkDeviceSize offset) const { return alignUp(offset, resourceHeapAlignment_); }

    VkResult writeSamplerDescriptor(const VkSamplerCreateInfo& samplerCreateInfo, void* dst) const
    {
        if (!initialized() || dst == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        const VkHostAddressRangeEXT dstRange{
            .address = dst,
            .size = static_cast<size_t>(samplerDescriptorSize_),
        };
        return vkWriteSamplerDescriptorsEXT(device_, 1, &samplerCreateInfo, &dstRange);
    }

    VkResult writeSamplerDescriptors(
        uint32_t descriptorCount,
        const VkSamplerCreateInfo* samplerCreateInfos,
        const VkHostAddressRangeEXT* dstRanges) const
    {
        if (!initialized() || descriptorCount == 0 || samplerCreateInfos == nullptr || dstRanges == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return vkWriteSamplerDescriptorsEXT(
            device_,
            descriptorCount,
            samplerCreateInfos,
            dstRanges);
    }

    VkResult writeImageDescriptor(
        VkImage image,
        VkFormat format,
        VkImageLayout layout,
        const VkImageSubresourceRange& subresourceRange,
        VkImageViewType viewType,
        void* dst) const
    {
        if (!initialized() || dst == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = viewType,
            .format = format,
            .subresourceRange = subresourceRange,
        };
        VkImageDescriptorInfoEXT imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
            .pView = &viewInfo,
            .layout = layout,
        };
        VkResourceDescriptorInfoEXT resourceInfo{
            .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .data = {.pImage = &imageInfo},
        };
        const VkHostAddressRangeEXT dstRange{
            .address = dst,
            .size = static_cast<size_t>(imageDescriptorSize_),
        };
        return vkWriteResourceDescriptorsEXT(device_, 1, &resourceInfo, &dstRange);
    }

    VkResult writeResourceDescriptors(
        uint32_t descriptorCount,
        const VkResourceDescriptorInfoEXT* resourceInfos,
        const VkHostAddressRangeEXT* dstRanges) const
    {
        if (!initialized() || descriptorCount == 0 || resourceInfos == nullptr || dstRanges == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return vkWriteResourceDescriptorsEXT(device_, descriptorCount, resourceInfos, dstRanges);
    }

    VkResult writeBufferDescriptor(
        VkDeviceAddress bufferAddress,
        VkDeviceSize bufferSize,
        VkDescriptorType type,
        void* dst) const
    {
        if (!initialized() || dst == nullptr || bufferAddress == 0 || bufferSize == 0) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        if (type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        VkDeviceAddressRangeEXT addressRange{
            .address = bufferAddress,
            .size = bufferSize,
        };
        VkResourceDescriptorInfoEXT resourceInfo{
            .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
            .type = type,
            .data = {.pAddressRange = &addressRange},
        };
        const VkHostAddressRangeEXT dstRange{
            .address = dst,
            .size = static_cast<size_t>(bufferDescriptorSize_),
        };
        return vkWriteResourceDescriptorsEXT(device_, 1, &resourceInfo, &dstRange);
    }

    VkResult writeAccelerationStructureDescriptor(
        VkDeviceAddress accelerationStructureAddress,
        VkDeviceSize accelerationStructureSize,
        void* dst) const
    {
        if (!initialized() || dst == nullptr || accelerationStructureAddress == 0 ||
            accelerationStructureSize == 0) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        VkDeviceAddressRangeEXT addressRange{
            .address = accelerationStructureAddress,
            .size = accelerationStructureSize,
        };
        VkResourceDescriptorInfoEXT resourceInfo{
            .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .data = {.pAddressRange = &addressRange},
        };
        const VkHostAddressRangeEXT dstRange{
            .address = dst,
            .size = static_cast<size_t>(bufferDescriptorSize_),
        };
        return vkWriteResourceDescriptorsEXT(device_, 1, &resourceInfo, &dstRange);
    }

    VkResult writePartitionedAccelerationStructureDescriptor(
        VkDeviceAddress accelerationStructureAddress,
        VkDeviceSize accelerationStructureSize,
        void* dst) const
    {
        if (!initialized() || dst == nullptr || accelerationStructureAddress == 0 ||
            accelerationStructureSize == 0) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        VkDeviceAddressRangeEXT addressRange{
            .address = accelerationStructureAddress,
            .size = accelerationStructureSize,
        };
        VkResourceDescriptorInfoEXT resourceInfo{
            .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV,
            .data = {.pAddressRange = &addressRange},
        };
        const VkHostAddressRangeEXT dstRange{
            .address = dst,
            .size = static_cast<size_t>(bufferDescriptorSize_),
        };
        return vkWriteResourceDescriptorsEXT(device_, 1, &resourceInfo, &dstRange);
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDeviceSize samplerDescriptorSize_ = 0;
    VkDeviceSize imageDescriptorSize_ = 0;
    VkDeviceSize bufferDescriptorSize_ = 0;
    VkDeviceSize samplerDescriptorAlignment_ = 0;
    VkDeviceSize imageDescriptorAlignment_ = 0;
    VkDeviceSize bufferDescriptorAlignment_ = 0;
    VkDeviceSize samplerHeapAlignment_ = 0;
    VkDeviceSize resourceHeapAlignment_ = 0;
    VkDeviceSize maxSamplerHeapSize_ = 0;
    VkDeviceSize maxResourceHeapSize_ = 0;
    VkDeviceSize minSamplerHeapReservedRange_ = 0;
    VkDeviceSize minResourceHeapReservedRange_ = 0;
    VkDeviceSize maxPushDataSize_ = 0;
};

template <typename T>
void appendPNext(void**& tail, T& value)
{
    value.pNext = nullptr;
    *tail = &value;
    tail = &value.pNext;
}

struct VulkanExtensionSet {
    std::vector<VkExtensionProperties> properties;
    bool swapchain = false;
    bool descriptorHeap = false;
    bool shaderObject = false;
    bool accelerationStructure = false;
    bool deferredHostOperations = false;
    bool rayQuery = false;
    bool rayTracingPipeline = false;
    bool pipelineLibrary = false;
    bool pushDescriptor = false;
    bool aftermathDiagnosticCheckpoints = false;
    bool aftermathDiagnosticsConfig = false;
    bool streamlineBinaryImport = false;
    bool streamlineImageViewHandle = false;
#ifdef VK_EXT_mesh_shader
    bool meshShader = false;
#endif
#ifdef VK_NV_cluster_acceleration_structure
    bool clusterAccelerationStructure = false;
#endif
#ifdef VK_NV_partitioned_acceleration_structure
    bool partitionedAccelerationStructure = false;
#endif
#ifdef VK_NV_cooperative_vector
    bool cooperativeVector = false;
#endif

    static VulkanExtensionSet query(VkPhysicalDevice physicalDevice)
    {
        VulkanExtensionSet result;
        result.properties = enumerateDeviceExtensions(physicalDevice);
        result.swapchain = result.has(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        result.descriptorHeap = result.has(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
        result.shaderObject = result.has(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
        result.accelerationStructure = result.has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        result.deferredHostOperations = result.has(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        result.rayQuery = result.has(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        result.rayTracingPipeline = result.has(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        result.pipelineLibrary = result.has(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
        result.pushDescriptor = result.has(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
#if defined(VK_NV_device_diagnostic_checkpoints)
        result.aftermathDiagnosticCheckpoints = result.has(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
#endif
#if defined(VK_NV_device_diagnostics_config)
        result.aftermathDiagnosticsConfig = result.has(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
#endif
#ifdef VK_NVX_binary_import
        result.streamlineBinaryImport = result.has(VK_NVX_BINARY_IMPORT_EXTENSION_NAME);
#endif
#ifdef VK_NVX_image_view_handle
        result.streamlineImageViewHandle = result.has(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME);
#endif
#ifdef VK_EXT_mesh_shader
        result.meshShader = result.has(VK_EXT_MESH_SHADER_EXTENSION_NAME);
#endif
#ifdef VK_NV_cluster_acceleration_structure
        result.clusterAccelerationStructure = result.has(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME);
#endif
#ifdef VK_NV_partitioned_acceleration_structure
        result.partitionedAccelerationStructure =
            result.has(VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME);
#endif
#ifdef VK_NV_cooperative_vector
        result.cooperativeVector = result.has(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
#endif
        return result;
    }

    bool has(const char* extensionName) const
    {
        return hasName(properties, extensionName);
    }
};

struct VulkanDeviceFeatureRequest {
    bool bindlessDescriptorHeap = false;
    bool shaderObject = false;
    bool meshShader = false;
    bool taskShader = false;
    bool geometryShader = false;
    bool rayTracingAccelerationStructure = false;
    bool rayQuery = false;
    bool pushDescriptor = false;
    bool clusterAccelerationStructure = false;
    bool partitionedAccelerationStructure = false;
    bool streamline = false;
    bool aftermath = false;

    static VulkanDeviceFeatureRequest from(const DeviceDesc& desc)
    {
        return VulkanDeviceFeatureRequest{
            .bindlessDescriptorHeap = desc.enableBindlessDescriptorHeap,
            .shaderObject = desc.enableShaderObject,
            .meshShader = desc.enableMeshShader,
            .taskShader = desc.enableTaskShader,
            .geometryShader = desc.enableGeometryShader,
            .rayTracingAccelerationStructure = desc.enableRayTracingAccelerationStructure,
            .rayQuery = desc.enableRayQuery,
            .pushDescriptor = desc.enablePushDescriptor,
            .clusterAccelerationStructure = desc.enableClusterAccelerationStructure,
            .partitionedAccelerationStructure = desc.enablePartitionedAccelerationStructure,
            .streamline = desc.enableStreamline,
            .aftermath = desc.enableAftermath && profiling::nsightAftermathInitialized(),
        };
    }
};

struct VulkanDeviceFeatureProbe {
    VkPhysicalDeviceVulkan11Features vulkan11Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features vulkan12Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    VkPhysicalDeviceVulkan13Features vulkan13Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptorHeapFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
    };
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    };
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
    };
#if defined(VK_NV_device_diagnostics_config)
    VkPhysicalDeviceDiagnosticsConfigFeaturesNV diagnosticsConfigFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV,
    };
#endif
#ifdef VK_EXT_mesh_shader
    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
    };
#endif
#ifdef VK_NV_cluster_acceleration_structure
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clusterAccelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV,
    };
#endif
#ifdef VK_NV_partitioned_acceleration_structure
    VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV partitionedAccelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV,
    };
#endif
#ifdef VK_NV_cooperative_vector
    VkPhysicalDeviceCooperativeVectorFeaturesNV cooperativeVectorFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
    };
#endif
    VkPhysicalDeviceFeatures2 features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    };

    void query(VkPhysicalDevice physicalDevice, const VulkanExtensionSet& extensions)
    {
        features.pNext = &vulkan11Features;
        void** featureTail = &vulkan11Features.pNext;
        appendPNext(featureTail, vulkan12Features);
        appendPNext(featureTail, vulkan13Features);
        if (extensions.descriptorHeap) {
            appendPNext(featureTail, descriptorHeapFeatures);
        }
        if (extensions.shaderObject) {
            appendPNext(featureTail, shaderObjectFeatures);
        }
        if (extensions.accelerationStructure) {
            appendPNext(featureTail, accelerationStructureFeatures);
        }
        if (extensions.rayQuery) {
            appendPNext(featureTail, rayQueryFeatures);
        }
        if (extensions.rayTracingPipeline) {
            appendPNext(featureTail, rayTracingPipelineFeatures);
        }
#if defined(VK_NV_device_diagnostics_config)
        if (extensions.aftermathDiagnosticsConfig) {
            appendPNext(featureTail, diagnosticsConfigFeatures);
        }
#endif
#ifdef VK_EXT_mesh_shader
        if (extensions.meshShader) {
            appendPNext(featureTail, meshShaderFeatures);
        }
#endif
#ifdef VK_NV_cluster_acceleration_structure
        if (extensions.clusterAccelerationStructure) {
            appendPNext(featureTail, clusterAccelerationStructureFeatures);
        }
#endif
#ifdef VK_NV_partitioned_acceleration_structure
        if (extensions.partitionedAccelerationStructure) {
            appendPNext(featureTail, partitionedAccelerationStructureFeatures);
        }
#endif
#ifdef VK_NV_cooperative_vector
        if (extensions.cooperativeVector) {
            appendPNext(featureTail, cooperativeVectorFeatures);
        }
#endif
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
    }

    bool supportsRequiredCoreFeatures() const
    {
        return vulkan11Features.shaderDrawParameters == VK_TRUE &&
            vulkan12Features.timelineSemaphore == VK_TRUE &&
            vulkan13Features.dynamicRendering == VK_TRUE &&
            vulkan13Features.synchronization2 == VK_TRUE;
    }

    bool supportsAccelerationStructure(const VulkanExtensionSet& extensions) const
    {
        return extensions.accelerationStructure &&
            extensions.deferredHostOperations &&
            accelerationStructureFeatures.accelerationStructure == VK_TRUE &&
            vulkan12Features.bufferDeviceAddress == VK_TRUE;
    }

    bool supportsClusterAccelerationStructure(
        const VulkanExtensionSet& extensions,
        bool accelerationStructureSupported) const
    {
#ifdef VK_NV_cluster_acceleration_structure
        return accelerationStructureSupported &&
            extensions.clusterAccelerationStructure &&
            clusterAccelerationStructureFeatures.clusterAccelerationStructure == VK_TRUE;
#else
        (void)extensions;
        (void)accelerationStructureSupported;
        return false;
#endif
    }

    bool supportsPartitionedAccelerationStructure(
        const VulkanExtensionSet& extensions,
        bool accelerationStructureSupported) const
    {
#ifdef VK_NV_partitioned_acceleration_structure
        return accelerationStructureSupported &&
            extensions.partitionedAccelerationStructure &&
            partitionedAccelerationStructureFeatures.partitionedAccelerationStructure == VK_TRUE;
#else
        (void)extensions;
        (void)accelerationStructureSupported;
        return false;
#endif
    }

    bool supportsStreamline(const VulkanExtensionSet& extensions, bool accelerationStructureSupported) const
    {
        return accelerationStructureSupported &&
            extensions.rayQuery &&
            rayQueryFeatures.rayQuery == VK_TRUE &&
            extensions.rayTracingPipeline &&
            rayTracingPipelineFeatures.rayTracingPipeline == VK_TRUE &&
            extensions.pipelineLibrary &&
            extensions.pushDescriptor &&
            extensions.streamlineBinaryImport &&
            extensions.streamlineImageViewHandle;
    }

    bool supportsAftermath(const VulkanExtensionSet& extensions) const
    {
#if defined(VK_NV_device_diagnostic_checkpoints) && defined(VK_NV_device_diagnostics_config)
        return profiling::nsightAftermathInitialized() &&
            extensions.aftermathDiagnosticCheckpoints &&
            extensions.aftermathDiagnosticsConfig &&
            diagnosticsConfigFeatures.diagnosticsConfig == VK_TRUE;
#else
        (void)extensions;
        return false;
#endif
    }

    bool supportsMeshShader(const VulkanExtensionSet& extensions) const
    {
#ifdef VK_EXT_mesh_shader
        return extensions.meshShader && meshShaderFeatures.meshShader == VK_TRUE;
#else
        (void)extensions;
        return false;
#endif
    }

    bool supportsTaskShader(const VulkanExtensionSet& extensions) const
    {
#ifdef VK_EXT_mesh_shader
        return extensions.meshShader && meshShaderFeatures.taskShader == VK_TRUE;
#else
        (void)extensions;
        return false;
#endif
    }
};

struct VulkanDeviceFeatureSelection {
    bool shaderDemoteToHelperInvocation = false;
    bool shaderIntegerDotProduct = false;
    bool cooperativeVector = false;
    bool bindlessDescriptorHeap = false;
    bool shaderObject = false;
    bool meshShader = false;
    bool taskShader = false;
    bool geometryShader = false;
    bool rayTracingAccelerationStructure = false;
    bool rayQuery = false;
    bool pushDescriptor = false;
    bool clusterAccelerationStructure = false;
    bool partitionedAccelerationStructure = false;
    bool streamline = false;
    bool aftermath = false;
    // Vulkan 1.2 core features required by the NRC SDK (scalar/standard layouts,
    // fp16/int16 shader capabilities) and by SHaRC's 64-bit hash-grid atomics.
    // Enabled opportunistically.
    bool scalarBlockLayout = false;
    bool uniformBufferStandardLayout = false;
    bool shaderBufferInt64Atomics = false;
    bool shaderFloat16 = false;
    bool shaderInt16 = false;
    // Extension availability used by enabledDeviceExtensions().
    bool nvxBinaryImport = false;
    bool nvxImageViewHandle = false;

    static VulkanDeviceFeatureSelection select(
        const VulkanDeviceFeatureRequest& request,
        VkPhysicalDevice physicalDevice,
        const VulkanExtensionSet& extensions,
        const VulkanDeviceFeatureProbe& probe)
    {
        const bool accelerationStructureSupported = probe.supportsAccelerationStructure(extensions);
        const bool clusterAccelerationStructureSupported =
            probe.supportsClusterAccelerationStructure(extensions, accelerationStructureSupported);
        const bool partitionedAccelerationStructureSupported =
            probe.supportsPartitionedAccelerationStructure(extensions, accelerationStructureSupported);
        const bool streamlineSupported = probe.supportsStreamline(extensions, accelerationStructureSupported);
        const bool aftermathSupported = probe.supportsAftermath(extensions);
        const bool meshShaderSupported = probe.supportsMeshShader(extensions);
        const bool taskShaderSupported = probe.supportsTaskShader(extensions);

        VulkanDeviceFeatureSelection result;
        result.shaderDemoteToHelperInvocation =
            probe.vulkan13Features.shaderDemoteToHelperInvocation == VK_TRUE;
        result.shaderIntegerDotProduct =
            probe.vulkan13Features.shaderIntegerDotProduct == VK_TRUE;
#ifdef VK_NV_cooperative_vector
        result.cooperativeVector =
            extensions.cooperativeVector &&
            probe.cooperativeVectorFeatures.cooperativeVector == VK_TRUE &&
            probe.vulkan12Features.bufferDeviceAddress == VK_TRUE;
#endif
        result.bindlessDescriptorHeap =
            request.bindlessDescriptorHeap &&
            extensions.descriptorHeap &&
            probe.descriptorHeapFeatures.descriptorHeap == VK_TRUE &&
            probe.vulkan12Features.descriptorIndexing == VK_TRUE &&
            probe.vulkan12Features.runtimeDescriptorArray == VK_TRUE &&
            probe.vulkan12Features.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
            probe.vulkan12Features.bufferDeviceAddress == VK_TRUE &&
            DescriptorHeapWriter::hasUsableProperties(physicalDevice);
        result.shaderObject =
            request.shaderObject &&
            extensions.shaderObject &&
            probe.shaderObjectFeatures.shaderObject == VK_TRUE;
        result.meshShader = request.meshShader && meshShaderSupported;
        result.taskShader = request.taskShader && taskShaderSupported;
        result.geometryShader =
            request.geometryShader && probe.features.features.geometryShader == VK_TRUE;
        result.rayTracingAccelerationStructure =
            (request.rayTracingAccelerationStructure ||
                request.rayQuery ||
                request.clusterAccelerationStructure ||
                request.partitionedAccelerationStructure ||
                request.streamline) &&
            accelerationStructureSupported;
        result.rayQuery =
            (request.rayQuery || request.streamline) &&
            accelerationStructureSupported &&
            extensions.rayQuery &&
            probe.rayQueryFeatures.rayQuery == VK_TRUE;
        result.pushDescriptor = (request.pushDescriptor || request.streamline) && extensions.pushDescriptor;
        result.clusterAccelerationStructure =
            request.clusterAccelerationStructure &&
            clusterAccelerationStructureSupported &&
            result.rayTracingAccelerationStructure;
        result.partitionedAccelerationStructure =
            request.partitionedAccelerationStructure &&
            partitionedAccelerationStructureSupported &&
            result.rayTracingAccelerationStructure;
        result.streamline =
            request.streamline &&
            streamlineSupported &&
            result.rayTracingAccelerationStructure &&
            result.rayQuery &&
            result.pushDescriptor;
        result.aftermath = request.aftermath && aftermathSupported;
        result.scalarBlockLayout = probe.vulkan12Features.scalarBlockLayout == VK_TRUE;
        result.uniformBufferStandardLayout = probe.vulkan12Features.uniformBufferStandardLayout == VK_TRUE;
        result.shaderBufferInt64Atomics = probe.vulkan12Features.shaderBufferInt64Atomics == VK_TRUE;
        result.shaderFloat16 = probe.vulkan12Features.shaderFloat16 == VK_TRUE;
        result.shaderInt16 = probe.features.features.shaderInt16 == VK_TRUE;
        result.nvxBinaryImport = extensions.streamlineBinaryImport;
        result.nvxImageViewHandle = extensions.streamlineImageViewHandle;
        return result;
    }

    bool usesBufferDeviceAddress() const
    {
        return bindlessDescriptorHeap ||
            cooperativeVector ||
            rayTracingAccelerationStructure ||
            rayQuery ||
            clusterAccelerationStructure ||
            partitionedAccelerationStructure ||
            streamline;
    }

    bool matches(const VulkanDeviceFeatureRequest& request) const
    {
        return (!request.bindlessDescriptorHeap || bindlessDescriptorHeap) &&
            (!request.shaderObject || shaderObject) &&
            (!request.meshShader || meshShader) &&
            (!request.taskShader || taskShader) &&
            (!request.geometryShader || geometryShader) &&
            (!request.rayTracingAccelerationStructure || rayTracingAccelerationStructure) &&
            (!request.rayQuery || rayQuery) &&
            (!request.pushDescriptor || pushDescriptor) &&
            (!request.clusterAccelerationStructure || clusterAccelerationStructure) &&
            (!request.partitionedAccelerationStructure || partitionedAccelerationStructure);
    }

    int32_t score() const
    {
        return (bindlessDescriptorHeap ? 16 : 0) +
            (cooperativeVector ? 16 : 0) +
            (partitionedAccelerationStructure ? 128 : 0) +
            (clusterAccelerationStructure ? 64 : 0) +
            (streamline ? 32 : 0) +
            (shaderObject ? 8 : 0) +
            (meshShader ? 8 : 0) +
            (taskShader ? 8 : 0) +
            (rayTracingAccelerationStructure ? 4 : 0) +
            (rayQuery ? 2 : 0) +
            (pushDescriptor ? 1 : 0);
    }
};

struct VulkanEnabledFeatureChain {
    VkPhysicalDeviceVulkan11Features vulkan11Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features vulkan12Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    VkPhysicalDeviceVulkan13Features vulkan13Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptorHeapFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
    };
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    };
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
    };
#ifdef VK_EXT_mesh_shader
    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
    };
#endif
#ifdef VK_NV_cluster_acceleration_structure
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clusterAccelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV,
    };
#endif
#ifdef VK_NV_partitioned_acceleration_structure
    VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV partitionedAccelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV,
    };
#endif
#ifdef VK_NV_cooperative_vector
    VkPhysicalDeviceCooperativeVectorFeaturesNV cooperativeVectorFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
    };
#endif
#if defined(VK_NV_device_diagnostics_config)
    VkPhysicalDeviceDiagnosticsConfigFeaturesNV diagnosticsConfigFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV,
    };
    VkDeviceDiagnosticsConfigCreateInfoNV diagnosticsConfigCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
        .flags =
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV,
    };
#endif
    VkPhysicalDeviceFeatures2 features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    };

    explicit VulkanEnabledFeatureChain(const VulkanDeviceFeatureSelection& selection)
    {
        vulkan11Features.shaderDrawParameters = VK_TRUE;
        vulkan12Features.descriptorIndexing = selection.bindlessDescriptorHeap ? VK_TRUE : VK_FALSE;
        vulkan12Features.shaderSampledImageArrayNonUniformIndexing =
            selection.bindlessDescriptorHeap ? VK_TRUE : VK_FALSE;
        vulkan12Features.runtimeDescriptorArray = selection.bindlessDescriptorHeap ? VK_TRUE : VK_FALSE;
        vulkan12Features.bufferDeviceAddress = selection.usesBufferDeviceAddress() ? VK_TRUE : VK_FALSE;
        vulkan12Features.timelineSemaphore = VK_TRUE;
        vulkan12Features.scalarBlockLayout = selection.scalarBlockLayout ? VK_TRUE : VK_FALSE;
        vulkan12Features.uniformBufferStandardLayout = selection.uniformBufferStandardLayout ? VK_TRUE : VK_FALSE;
        vulkan12Features.shaderBufferInt64Atomics = selection.shaderBufferInt64Atomics ? VK_TRUE : VK_FALSE;
        vulkan12Features.shaderFloat16 = selection.shaderFloat16 ? VK_TRUE : VK_FALSE;
        features.features.shaderInt16 = selection.shaderInt16 ? VK_TRUE : VK_FALSE;
        features.features.geometryShader = selection.geometryShader ? VK_TRUE : VK_FALSE;
        vulkan13Features.synchronization2 = VK_TRUE;
        vulkan13Features.dynamicRendering = VK_TRUE;
        vulkan13Features.shaderDemoteToHelperInvocation =
            selection.shaderDemoteToHelperInvocation ? VK_TRUE : VK_FALSE;
        vulkan13Features.shaderIntegerDotProduct =
            selection.shaderIntegerDotProduct ? VK_TRUE : VK_FALSE;
#ifdef VK_NV_cooperative_vector
        cooperativeVectorFeatures.cooperativeVector =
            selection.cooperativeVector ? VK_TRUE : VK_FALSE;
#endif
        descriptorHeapFeatures.descriptorHeap = selection.bindlessDescriptorHeap ? VK_TRUE : VK_FALSE;
        shaderObjectFeatures.shaderObject = selection.shaderObject ? VK_TRUE : VK_FALSE;
#ifdef VK_EXT_mesh_shader
        meshShaderFeatures.meshShader = selection.meshShader ? VK_TRUE : VK_FALSE;
        meshShaderFeatures.taskShader = selection.taskShader ? VK_TRUE : VK_FALSE;
#endif
        accelerationStructureFeatures.accelerationStructure =
            selection.rayTracingAccelerationStructure ? VK_TRUE : VK_FALSE;
        rayQueryFeatures.rayQuery = selection.rayQuery ? VK_TRUE : VK_FALSE;
        rayTracingPipelineFeatures.rayTracingPipeline = selection.streamline ? VK_TRUE : VK_FALSE;
#ifdef VK_NV_cluster_acceleration_structure
        clusterAccelerationStructureFeatures.clusterAccelerationStructure =
            selection.clusterAccelerationStructure ? VK_TRUE : VK_FALSE;
#endif
#ifdef VK_NV_partitioned_acceleration_structure
        partitionedAccelerationStructureFeatures.partitionedAccelerationStructure =
            selection.partitionedAccelerationStructure ? VK_TRUE : VK_FALSE;
#endif
#if defined(VK_NV_device_diagnostics_config)
        diagnosticsConfigFeatures.diagnosticsConfig = selection.aftermath ? VK_TRUE : VK_FALSE;
#endif

        features.pNext = &vulkan11Features;
        void** featureTail = &vulkan11Features.pNext;
        appendPNext(featureTail, vulkan12Features);
        appendPNext(featureTail, vulkan13Features);
        if (selection.bindlessDescriptorHeap) {
            appendPNext(featureTail, descriptorHeapFeatures);
        }
        if (selection.shaderObject) {
            appendPNext(featureTail, shaderObjectFeatures);
        }
#ifdef VK_EXT_mesh_shader
        if (selection.meshShader || selection.taskShader) {
            appendPNext(featureTail, meshShaderFeatures);
        }
#endif
        if (selection.rayTracingAccelerationStructure) {
            appendPNext(featureTail, accelerationStructureFeatures);
        }
        if (selection.rayQuery) {
            appendPNext(featureTail, rayQueryFeatures);
        }
        if (selection.streamline) {
            appendPNext(featureTail, rayTracingPipelineFeatures);
        }
#ifdef VK_NV_cluster_acceleration_structure
        if (selection.clusterAccelerationStructure) {
            appendPNext(featureTail, clusterAccelerationStructureFeatures);
        }
#endif
#ifdef VK_NV_partitioned_acceleration_structure
        if (selection.partitionedAccelerationStructure) {
            appendPNext(featureTail, partitionedAccelerationStructureFeatures);
        }
#endif
#ifdef VK_NV_cooperative_vector
        if (selection.cooperativeVector) {
            appendPNext(featureTail, cooperativeVectorFeatures);
        }
#endif
#if defined(VK_NV_device_diagnostics_config)
        if (selection.aftermath) {
            appendPNext(featureTail, diagnosticsConfigFeatures);
            diagnosticsConfigCreateInfo.pNext = nullptr;
            *featureTail = &diagnosticsConfigCreateInfo;
        }
#endif
    }
};

std::vector<const char*> enabledDeviceExtensions(const VulkanDeviceFeatureSelection& selection)
{
    std::vector<const char*> extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    if (selection.bindlessDescriptorHeap) {
        extensions.push_back(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    }
    if (selection.shaderObject) {
        extensions.push_back(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
    }
#ifdef VK_EXT_mesh_shader
    if (selection.meshShader || selection.taskShader) {
        extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    }
#endif
    if (selection.rayTracingAccelerationStructure) {
        extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    }
    if (selection.rayQuery) {
        extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }
    if (selection.pushDescriptor) {
        extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    }
#ifdef VK_NV_cluster_acceleration_structure
    if (selection.clusterAccelerationStructure) {
        extensions.push_back(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    }
#endif
#ifdef VK_NV_partitioned_acceleration_structure
    if (selection.partitionedAccelerationStructure) {
        extensions.push_back(VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    }
#endif
#ifdef VK_NV_cooperative_vector
    if (selection.cooperativeVector) {
        extensions.push_back(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
    }
#endif
    if (selection.streamline) {
        extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
#ifdef VK_NVX_binary_import
        extensions.push_back(VK_NVX_BINARY_IMPORT_EXTENSION_NAME);
#endif
#ifdef VK_NVX_image_view_handle
        extensions.push_back(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME);
#endif
    }
    if (selection.aftermath) {
#if defined(VK_NV_device_diagnostic_checkpoints)
        extensions.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
#endif
#if defined(VK_NV_device_diagnostics_config)
        extensions.push_back(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
#endif
    }
    if (selection.scalarBlockLayout) {
        // The NRC SDK requires these extension names to be enabled even where
        // the functionality is core in Vulkan 1.2.
        extensions.push_back(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
        extensions.push_back(VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME);
    }
    if (selection.usesBufferDeviceAddress()) {
        // The NRC SDK checks for the pre-promotion extension name as well.
        extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        // NRC's device setup expects these NVIDIA kernel-launch extensions to
        // be enabled; they are inert unless explicitly used.
#ifdef VK_NVX_binary_import
        if (selection.nvxBinaryImport) {
            extensions.push_back(VK_NVX_BINARY_IMPORT_EXTENSION_NAME);
        }
#endif
#ifdef VK_NVX_image_view_handle
        if (selection.nvxImageViewHandle) {
            extensions.push_back(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME);
        }
#endif
    }
    return extensions;
}

struct VulkanPhysicalDeviceCandidate {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t graphicsFamily = 0;
    uint32_t computeFamily = 0;
    uint32_t copyFamily = UINT32_MAX;
    VulkanDeviceFeatureSelection features;
    int32_t featureScore = -1;
};

class DescriptorHeap {
public:
    VkResult initialize(VkPhysicalDevice physicalDevice, VkDevice device)
    {
        *this = {};
        const VkResult result = writer_.initialize(physicalDevice, device);
        if (result != VK_SUCCESS) {
            return result;
        }

        const VkDeviceSize samplerAvailable =
            writer_.maxSamplerHeapSize() > writer_.minSamplerHeapReservedRange()
            ? writer_.maxSamplerHeapSize() - writer_.minSamplerHeapReservedRange()
            : 0;
        const VkDeviceSize resourceAvailable =
            writer_.maxResourceHeapSize() > writer_.minResourceHeapReservedRange()
            ? writer_.maxResourceHeapSize() - writer_.minResourceHeapReservedRange()
            : 0;
        maxSamplerCapacity_ = capacityFromBytes(samplerAvailable, writer_.samplerDescriptorSize());
        maxImageCapacity_ = capacityFromBytes(resourceAvailable, writer_.imageDescriptorSize());
        maxBufferCapacity_ = capacityFromBytes(resourceAvailable, writer_.bufferDescriptorSize());
        return VK_SUCCESS;
    }

    bool initialized() const { return writer_.initialized(); }
    const DescriptorHeapWriter& writer() const { return writer_; }
    uint32_t maxSamplerCapacity() const { return maxSamplerCapacity_; }
    uint32_t maxImageCapacity() const { return maxImageCapacity_; }
    uint32_t maxBufferCapacity() const { return maxBufferCapacity_; }
    uint32_t maxImages() const { return maxImages_; }
    uint32_t maxBuffers() const { return maxBuffers_; }
    VkDeviceSize samplerHeapSize() const { return samplerHeapSize_; }
    VkDeviceSize resourceHeapSize() const { return resourceHeapSize_; }
    VkDeviceSize samplerHeapAlignment() const { return writer_.samplerHeapAlignment(); }
    VkDeviceSize resourceHeapAlignment() const { return writer_.resourceHeapAlignment(); }

    VkDeviceSize setupSamplerHeap(uint32_t maxSamplers)
    {
        if (!initialized() || maxSamplers == 0 || maxSamplers > maxSamplerCapacity_) {
            return 0;
        }

        VkDeviceSize offset = 0;
        writer_.appendSamplerDescriptors(offset, maxSamplers);
        writer_.appendSamplerReservedRange(offset);
        samplerHeapSize_ = writer_.alignToSamplerHeap(offset);
        maxSamplers_ = maxSamplers;
        nextSamplerSlot_ = 0;
        freeSamplerSlots_.clear();
        clearSamplerDirty();
        return samplerHeapSize_;
    }

    VkDeviceSize setupResourceHeap(uint32_t maxImages, uint32_t maxBuffers)
    {
        if (!initialized() || (maxImages == 0 && maxBuffers == 0)) {
            return 0;
        }

        VkDeviceSize offset = 0;
        imageRegionStartBytes_ = writer_.appendImageDescriptors(offset, maxImages);
        bufferRegionStartBytes_ = writer_.appendBufferDescriptors(offset, maxBuffers);
        resourceReservedRangeOffsetBytes_ = writer_.appendResourceReservedRange(offset);
        const VkDeviceSize packedSize = writer_.alignToResourceHeap(offset);
        if (packedSize > writer_.maxResourceHeapSize()) {
            return 0;
        }

        resourceHeapSize_ = packedSize;
        maxImages_ = maxImages;
        maxBuffers_ = maxBuffers;
        nextImageSlot_ = 0;
        nextBufferSlot_ = 0;
        freeImageSlots_.clear();
        freeBufferSlots_.clear();
        clearResourceDirty();
        return resourceHeapSize_;
    }

    uint32_t imageShaderIndexBase() const
    {
        const VkDeviceSize size = writer_.imageDescriptorSize();
        return size > 0 ? static_cast<uint32_t>(imageRegionStartBytes_ / size) : 0;
    }

    uint32_t bufferShaderIndexBase() const
    {
        const VkDeviceSize size = writer_.bufferDescriptorSize();
        return size > 0 ? static_cast<uint32_t>(bufferRegionStartBytes_ / size) : 0;
    }

    bool allocateSampler(BindlessHandle& outHandle)
    {
        uint32_t slot = 0;
        if (!allocateSlot(maxSamplers_, nextSamplerSlot_, freeSamplerSlots_, slot)) {
            return false;
        }
        outHandle = {
            .kind = BindlessHandleKind::Sampler,
            .index = slot,
            .shaderIndex = slot,
        };
        return true;
    }

    bool allocateSampledImage(BindlessHandle& outHandle)
    {
        uint32_t slot = 0;
        if (!allocateSlot(maxImages_, nextImageSlot_, freeImageSlots_, slot)) {
            return false;
        }
        outHandle = {
            .kind = BindlessHandleKind::SampledImage,
            .index = slot,
            .shaderIndex = imageShaderIndexBase() + slot,
        };
        return true;
    }

    bool allocateStorageImage(BindlessHandle& outHandle)
    {
        uint32_t slot = 0;
        if (!allocateSlot(maxImages_, nextImageSlot_, freeImageSlots_, slot)) {
            return false;
        }
        outHandle = {
            .kind = BindlessHandleKind::StorageImage,
            .index = slot,
            .shaderIndex = imageShaderIndexBase() + slot,
        };
        return true;
    }

    bool allocateBuffer(BindlessHandle& outHandle)
    {
        uint32_t slot = 0;
        if (!allocateSlot(maxBuffers_, nextBufferSlot_, freeBufferSlots_, slot)) {
            return false;
        }
        outHandle = {
            .kind = BindlessHandleKind::Buffer,
            .index = slot,
            .shaderIndex = bufferShaderIndexBase() + slot,
        };
        return true;
    }

    void release(BindlessHandle handle)
    {
        if (!handle.valid()) {
            return;
        }
        switch (handle.kind) {
        case BindlessHandleKind::SampledImage:
        case BindlessHandleKind::StorageImage:
            if (handle.index < maxImages_) {
                freeImageSlots_.push_back(handle.index);
            }
            break;
        case BindlessHandleKind::Buffer:
        case BindlessHandleKind::AccelerationStructure:
        case BindlessHandleKind::PartitionedAccelerationStructure:
            if (handle.index < maxBuffers_) {
                freeBufferSlots_.push_back(handle.index);
            }
            break;
        case BindlessHandleKind::Sampler:
            if (handle.index < maxSamplers_) {
                freeSamplerSlots_.push_back(handle.index);
            }
            break;
        case BindlessHandleKind::Invalid:
            break;
        }
    }

    VkResult writeSamplerDescriptors(
        const BindlessHandle* handles,
        const VkSamplerCreateInfo* samplerInfos,
        uint32_t descriptorCount,
        void* samplerHeapBase)
    {
        if (handles == nullptr || samplerInfos == nullptr || descriptorCount == 0 || samplerHeapBase == nullptr) {
            return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        std::vector<VkHostAddressRangeEXT> dstRanges(descriptorCount);
        for (uint32_t index = 0; index < descriptorCount; ++index) {
            const BindlessHandle handle = handles[index];
            if (handle.kind != BindlessHandleKind::Sampler || handle.index >= maxSamplers_) {
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
            dstRanges[index] = {
                .address = static_cast<uint8_t*>(samplerHeapBase) + writer_.samplerOffset(handle.index),
                .size = static_cast<size_t>(writer_.samplerDescriptorSize()),
            };
        }

        const VkResult result = writer_.writeSamplerDescriptors(
            descriptorCount,
            samplerInfos,
            dstRanges.data());
        if (result == VK_SUCCESS) {
            for (uint32_t index = 0; index < descriptorCount; ++index) {
                samplerDirtyMin_ = std::min(samplerDirtyMin_, handles[index].index);
                samplerDirtyMax_ = std::max(samplerDirtyMax_, handles[index].index);
            }
        }
        return result;
    }

    VkResult writeImageDescriptors(
        const BindlessHandle* handles,
        const VkResourceDescriptorInfoEXT* resourceInfos,
        uint32_t descriptorCount,
        void* resourceHeapBase)
    {
        if (handles == nullptr || resourceInfos == nullptr || descriptorCount == 0 || resourceHeapBase == nullptr) {
            return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        std::vector<VkHostAddressRangeEXT> dstRanges(descriptorCount);
        for (uint32_t index = 0; index < descriptorCount; ++index) {
            const BindlessHandle handle = handles[index];
            const bool validKind = handle.kind == BindlessHandleKind::SampledImage ||
                handle.kind == BindlessHandleKind::StorageImage;
            if (!validKind || handle.index >= maxImages_) {
                return VK_ERROR_VALIDATION_FAILED_EXT;
            }
            const VkDeviceSize offset = imageRegionStartBytes_ + writer_.imageOffset(handle.index);
            dstRanges[index] = {
                .address = static_cast<uint8_t*>(resourceHeapBase) + offset,
                .size = static_cast<size_t>(writer_.imageDescriptorSize()),
            };
        }

        const VkResult result = writer_.writeResourceDescriptors(
            descriptorCount,
            resourceInfos,
            dstRanges.data());
        if (result == VK_SUCCESS) {
            for (uint32_t index = 0; index < descriptorCount; ++index) {
                const VkDeviceSize offset = imageRegionStartBytes_ + writer_.imageOffset(handles[index].index);
                markResourceImageDirty(offset, writer_.imageDescriptorSize());
            }
        }
        return result;
    }

    VkResult writeImageDescriptor(
        BindlessHandle handle,
        VkImage image,
        VkFormat format,
        VkImageLayout layout,
        const VkImageSubresourceRange& subresourceRange,
        VkImageViewType viewType,
        void* resourceHeapBase)
    {
        if ((handle.kind != BindlessHandleKind::SampledImage &&
             handle.kind != BindlessHandleKind::StorageImage) ||
            handle.index >= maxImages_ ||
            resourceHeapBase == nullptr) {
            return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        const VkDeviceSize descriptorSize = writer_.imageDescriptorSize();
        const VkDeviceSize offset = imageRegionStartBytes_ + writer_.imageOffset(handle.index);
        void* dst = static_cast<uint8_t*>(resourceHeapBase) + offset;
        const VkResult result = writer_.writeImageDescriptor(
            image,
            format,
            layout,
            subresourceRange,
            viewType,
            dst);
        if (result == VK_SUCCESS) {
            markResourceImageDirty(offset, descriptorSize);
        }
        return result;
    }

    VkResult writeBufferDescriptor(
        BindlessHandle handle,
        VkDeviceAddress address,
        VkDeviceSize size,
        VkDescriptorType type,
        void* resourceHeapBase)
    {
        if (handle.kind != BindlessHandleKind::Buffer ||
            handle.index >= maxBuffers_ ||
            resourceHeapBase == nullptr) {
            return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        const VkDeviceSize descriptorSize = writer_.bufferDescriptorSize();
        const VkDeviceSize offset = bufferRegionStartBytes_ + writer_.bufferOffset(handle.index);
        void* dst = static_cast<uint8_t*>(resourceHeapBase) + offset;
        const VkResult result = writer_.writeBufferDescriptor(address, size, type, dst);
        if (result == VK_SUCCESS) {
            markResourceBufferDirty(offset, descriptorSize);
        }
        return result;
    }

    VkResult writeAccelerationStructureDescriptor(
        BindlessHandle handle,
        VkDeviceAddress address,
        VkDeviceSize size,
        void* resourceHeapBase)
    {
        if (handle.kind != BindlessHandleKind::AccelerationStructure ||
            handle.index >= maxBuffers_ || resourceHeapBase == nullptr) {
            return VK_ERROR_VALIDATION_FAILED_EXT;
        }

        const VkDeviceSize descriptorSize = writer_.bufferDescriptorSize();
        const VkDeviceSize offset = bufferRegionStartBytes_ + writer_.bufferOffset(handle.index);
        void* dst = static_cast<uint8_t*>(resourceHeapBase) + offset;
        const VkResult result = writer_.writeAccelerationStructureDescriptor(address, size, dst);
        if (result == VK_SUCCESS) {
            markResourceBufferDirty(offset, descriptorSize);
        }
        return result;
    }

    VkResult writePartitionedAccelerationStructureDescriptor(
        BindlessHandle handle,
        VkDeviceAddress address,
        VkDeviceSize size,
        void* resourceHeapBase)
    {
        if (handle.kind != BindlessHandleKind::PartitionedAccelerationStructure ||
            handle.index >= maxBuffers_ || resourceHeapBase == nullptr) {
            return VK_ERROR_VALIDATION_FAILED_EXT;
        }
        const VkDeviceSize descriptorSize = writer_.bufferDescriptorSize();
        const VkDeviceSize offset = bufferRegionStartBytes_ + writer_.bufferOffset(handle.index);
        void* dst = static_cast<uint8_t*>(resourceHeapBase) + offset;
        const VkResult result = writer_.writePartitionedAccelerationStructureDescriptor(
            address,
            size,
            dst);
        if (result == VK_SUCCESS) {
            markResourceBufferDirty(offset, descriptorSize);
        }
        return result;
    }

    struct DirtyRange {
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
    };

    DirtyRange samplerDirtyRange() const
    {
        if (samplerDirtyMin_ > samplerDirtyMax_) {
            return {};
        }
        const VkDeviceSize descriptorSize = writer_.samplerDescriptorSize();
        return {
            .offset = static_cast<VkDeviceSize>(samplerDirtyMin_) * descriptorSize,
            .size = static_cast<VkDeviceSize>(samplerDirtyMax_ - samplerDirtyMin_ + 1) * descriptorSize,
        };
    }

    DirtyRange resourceImageDirtyRange() const
    {
        if (resourceImageDirtyMin_ > resourceImageDirtyMax_) {
            return {};
        }
        return {
            .offset = resourceImageDirtyMin_,
            .size = resourceImageDirtyMax_ - resourceImageDirtyMin_ + 1,
        };
    }

    DirtyRange resourceBufferDirtyRange() const
    {
        if (resourceBufferDirtyMin_ > resourceBufferDirtyMax_) {
            return {};
        }
        return {
            .offset = resourceBufferDirtyMin_,
            .size = resourceBufferDirtyMax_ - resourceBufferDirtyMin_ + 1,
        };
    }

    void clearSamplerDirty()
    {
        samplerDirtyMin_ = std::numeric_limits<uint32_t>::max();
        samplerDirtyMax_ = 0;
    }

    void clearResourceDirty()
    {
        resourceImageDirtyMin_ = std::numeric_limits<VkDeviceSize>::max();
        resourceImageDirtyMax_ = 0;
        resourceBufferDirtyMin_ = std::numeric_limits<VkDeviceSize>::max();
        resourceBufferDirtyMax_ = 0;
    }

    void bind(VkCommandBuffer commandBuffer, VkDeviceAddress samplerHeapAddress, VkDeviceAddress resourceHeapAddress) const
    {
        if (samplerHeapAddress != 0 && maxSamplers_ > 0) {
            const VkBindHeapInfoEXT samplerBind{
                .sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
                .heapRange = {
                    .address = samplerHeapAddress,
                    .size = samplerHeapSize_,
                },
                .reservedRangeOffset = writer_.samplerDescriptorSize() * maxSamplers_,
                .reservedRangeSize = writer_.minSamplerHeapReservedRange(),
            };
            vkCmdBindSamplerHeapEXT(commandBuffer, &samplerBind);
        }

        if (resourceHeapAddress != 0 && (maxImages_ > 0 || maxBuffers_ > 0)) {
            const VkBindHeapInfoEXT resourceBind{
                .sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
                .heapRange = {
                    .address = resourceHeapAddress,
                    .size = resourceHeapSize_,
                },
                .reservedRangeOffset = resourceReservedRangeOffsetBytes_,
                .reservedRangeSize = writer_.minResourceHeapReservedRange(),
            };
            vkCmdBindResourceHeapEXT(commandBuffer, &resourceBind);
        }
    }

private:
    static bool allocateSlot(uint32_t maxSlots, uint32_t& nextSlot, std::vector<uint32_t>& freeSlots, uint32_t& outSlot)
    {
        if (!freeSlots.empty()) {
            outSlot = freeSlots.back();
            freeSlots.pop_back();
            return true;
        }
        if (nextSlot >= maxSlots) {
            return false;
        }
        outSlot = nextSlot++;
        return true;
    }

    void markResourceImageDirty(VkDeviceSize offset, VkDeviceSize size)
    {
        resourceImageDirtyMin_ = std::min(resourceImageDirtyMin_, offset);
        resourceImageDirtyMax_ = std::max(resourceImageDirtyMax_, offset + size - 1);
    }

    void markResourceBufferDirty(VkDeviceSize offset, VkDeviceSize size)
    {
        resourceBufferDirtyMin_ = std::min(resourceBufferDirtyMin_, offset);
        resourceBufferDirtyMax_ = std::max(resourceBufferDirtyMax_, offset + size - 1);
    }

    DescriptorHeapWriter writer_;
    uint32_t maxSamplerCapacity_ = 0;
    uint32_t maxImageCapacity_ = 0;
    uint32_t maxBufferCapacity_ = 0;
    uint32_t maxSamplers_ = 0;
    uint32_t maxImages_ = 0;
    uint32_t maxBuffers_ = 0;
    VkDeviceSize samplerHeapSize_ = 0;
    VkDeviceSize resourceHeapSize_ = 0;
    VkDeviceSize imageRegionStartBytes_ = 0;
    VkDeviceSize bufferRegionStartBytes_ = 0;
    VkDeviceSize resourceReservedRangeOffsetBytes_ = 0;
    uint32_t nextSamplerSlot_ = 0;
    uint32_t nextImageSlot_ = 0;
    uint32_t nextBufferSlot_ = 0;
    std::vector<uint32_t> freeSamplerSlots_;
    std::vector<uint32_t> freeImageSlots_;
    std::vector<uint32_t> freeBufferSlots_;
    uint32_t samplerDirtyMin_ = std::numeric_limits<uint32_t>::max();
    uint32_t samplerDirtyMax_ = 0;
    VkDeviceSize resourceImageDirtyMin_ = std::numeric_limits<VkDeviceSize>::max();
    VkDeviceSize resourceImageDirtyMax_ = 0;
    VkDeviceSize resourceBufferDirtyMin_ = std::numeric_limits<VkDeviceSize>::max();
    VkDeviceSize resourceBufferDirtyMax_ = 0;
};

} // namespace

namespace detail {

struct DeviceImpl;

struct QueueImpl {
    DeviceImpl* device = nullptr;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = 0;
    uint32_t timestampValidBits = 0;
    QueueType type = QueueType::Graphics;
};

struct FenceImpl {
    DeviceImpl* device = nullptr;
    VkFence fence = VK_NULL_HANDLE;
};

struct TimestampQueryPoolImpl {
    DeviceImpl* device = nullptr;
    TimestampQueryPoolDesc desc;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    uint32_t timestampValidBits = 0;
    double timestampPeriodNanoseconds = 0.0;
};

struct SemaphoreImpl {
    DeviceImpl* device = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
};

struct SwapchainSemaphoreImpl {
    DeviceImpl* device = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
};

struct BufferImpl {
    DeviceImpl* device = nullptr;
    BufferDesc desc;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

struct RayTracingAccelerationStructureImpl {
    DeviceImpl* device = nullptr;
    RayTracingAccelerationStructureDesc desc;
    std::unique_ptr<Buffer> storage;
    VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;

    ~RayTracingAccelerationStructureImpl();
};

struct PartitionedAccelerationStructureImpl {
    DeviceImpl* device = nullptr;
    PartitionedAccelerationStructureDesc desc;
    std::unique_ptr<Buffer> storage;
    std::unique_ptr<Buffer> operationBuffer;
    std::unique_ptr<Buffer> operationCountBuffer;
    VkDeviceAddress address = 0;
};

struct BufferViewImpl {
    DeviceImpl* device = nullptr;
    Buffer* buffer = nullptr;
    BufferViewDesc desc;
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
};

struct TextureImpl {
    DeviceImpl* device = nullptr;
    TextureDesc desc;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageCreateFlags flags = 0;
    VkImageUsageFlags usage = 0;
    bool ownsImage = false;
};

struct TextureViewImpl {
    DeviceImpl* device = nullptr;
    Texture* texture = nullptr;
    TextureViewDesc desc;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct ShaderModuleImpl {
    DeviceImpl* device = nullptr;
    VkShaderModule module = VK_NULL_HANDLE;
    uint64_t contentHash = 0;
};

struct PipelineCacheImpl {
    DeviceImpl* device = nullptr;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::filesystem::path filePath;
    std::string filePathString;
    PipelineCacheFileIdentity fileIdentity;
    PipelineCacheStats stats;
    std::unordered_set<uint64_t> storedPsoHashes;
    std::unordered_set<uint64_t> sessionPsoHashes;
    mutable std::mutex mutex;
    bool saveOnDestroy = true;
    bool dirty = false;

    ~PipelineCacheImpl();
    Result initialize(DeviceImpl& owningDevice, const PipelineCacheDesc& desc);
    Result saveLocked();
    bool recordPsoLocked(uint64_t psoHash);
};

struct GraphicsPipelineImpl {
    DeviceImpl* device = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderStageFlags bindlessPushStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    bool usesBindlessHeap = false;
    uint64_t psoHash = 0;
    bool pipelineCacheHit = false;
};

struct ComputePipelineImpl {
    DeviceImpl* device = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool usesBindlessHeap = false;
    uint32_t bindlessUserDataOffset = sizeof(BindlessHeapPushConstants);
    uint64_t psoHash = 0;
    bool pipelineCacheHit = false;
};

struct GraphicsShaderObjectProgramImpl {
    DeviceImpl* device = nullptr;
    VkShaderEXT vertexShader = VK_NULL_HANDLE;
    VkShaderEXT fragmentShader = VK_NULL_HANDLE;
    bool usesBindlessHeap = false;
};

struct CommandPoolImpl {
    DeviceImpl* device = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
};

struct CommandBufferImpl {
    DeviceImpl* device = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkPipelineLayout currentGraphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout currentComputePipelineLayout = VK_NULL_HANDLE;
    VkShaderStageFlags currentGraphicsPipelineBindlessPushStages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    BindlessHeapImpl* currentBindlessHeap = nullptr;
    bool currentGraphicsPipelineUsesBindlessHeap = false;
    bool currentComputePipelineUsesBindlessHeap = false;
    bool currentGraphicsShaderObjectUsesBindlessHeap = false;
    bool currentGraphicsShaderObjectBound = false;
    uint32_t currentBindlessUserDataOffset = sizeof(BindlessHeapPushConstants);
    Viewport currentViewport;
    Rect currentScissor;
    bool hasCurrentViewport = false;
    bool hasCurrentScissor = false;
    std::vector<uint8_t> currentBindlessUserData;
};

struct SwapchainImpl {
    DeviceImpl* device = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat vkFormat = VK_FORMAT_UNDEFINED;
    Format format = Format::Unknown;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<std::unique_ptr<Texture>> textures;

    ~SwapchainImpl();
    Result initialize(const SwapchainDesc& desc);
    void wrapImages(const std::vector<VkImage>& images, TextureUsageBits usage);
};

struct BindlessHeapBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
    VkDeviceSize mappedOffset = 0;
};

struct BindlessHeapImpl {
    DeviceImpl* device = nullptr;
    BindlessHeapDesc desc;
    DescriptorHeap heap;
    BindlessHeapBuffer samplerHeap;
    BindlessHeapBuffer resourceHeap;

    ~BindlessHeapImpl();
    Result initialize(DeviceImpl& owningDevice, const BindlessHeapDesc& heapDesc);
    Result createHeapBuffer(VkDeviceSize size, VkDeviceSize alignment, BindlessHeapBuffer& outBuffer);
    void destroyHeapBuffer(BindlessHeapBuffer& buffer);
    void flushSamplerDirty();
    void flushResourceDirty();
};

struct DeviceImpl {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    DeviceCapabilities capabilities;
    PipelineCacheFileIdentity pipelineCacheFileIdentity;
    DescriptorHeapWriter descriptorHeapWriter;
    uint32_t graphicsFamily = 0;
    uint32_t computeFamily = 0;
    uint32_t copyFamily = UINT32_MAX;
    uint32_t copyQueueIndex = UINT32_MAX;
    SDL_SharedObject* vulkanLoaderHandle = nullptr;
    bool sdlVulkanLoaded = false;
    bool validationEnabled = false;
    bool debugUtilsEnabled = false;
    bool bindlessDescriptorHeapEnabled = false;
    bool shaderObjectEnabled = false;
    bool bufferDeviceAddressEnabled = false;
    bool rayTracingAccelerationStructureEnabled = false;
    bool rayQueryEnabled = false;
    bool pushDescriptorEnabled = false;
    bool clusterAccelerationStructureEnabled = false;
    bool partitionedAccelerationStructureEnabled = false;
    bool streamlineInitialized = false;
    PFN_vkSetDebugUtilsObjectNameEXT setDebugUtilsObjectName = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabel = nullptr;
    std::vector<std::unique_ptr<Queue>> queues;

    ~DeviceImpl();
    void addQueue(
        VkQueue queue,
        uint32_t familyIndex,
        uint32_t timestampValidBits,
        QueueType type);
};

RayTracingAccelerationStructureImpl::~RayTracingAccelerationStructureImpl()
{
    if (device != nullptr && accelerationStructure != VK_NULL_HANDLE) {
        activateVolkDevice(device->device);
        vkDestroyAccelerationStructureKHR(device->device, accelerationStructure, nullptr);
        accelerationStructure = VK_NULL_HANDLE;
    }
}

DeviceImpl::~DeviceImpl()
{
    queues.clear();

    if (device != VK_NULL_HANDLE) {
        activateVolkDevice(device);
        const VkResult waitResult = vkDeviceWaitIdle(device);
        if (waitResult == VK_ERROR_DEVICE_LOST) {
            profiling::handleNsightAftermathDeviceLost();
        }
    }

    if (streamlineInitialized) {
        vulkan::shutdownStreamline();
        streamlineInitialized = false;
    }

    if (allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE) {
        activateVolkDevice(device);
        vkDestroyDevice(device, nullptr);
        clearActiveVolkDevice(device);
        device = VK_NULL_HANDLE;
    }

    if (debugMessenger != VK_NULL_HANDLE) {
        destroyDebugMessenger(instance, debugMessenger);
        debugMessenger = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }

    if (sdlVulkanLoaded) {
        releaseSdlVulkanLibrary();
        sdlVulkanLoaded = false;
    }

    if (vulkanLoaderHandle != nullptr) {
        SDL_UnloadObject(vulkanLoaderHandle);
        vulkanLoaderHandle = nullptr;
    }
}

PipelineCacheImpl::~PipelineCacheImpl()
{
    if (device == nullptr || pipelineCache == VK_NULL_HANDLE) {
        return;
    }
    activateVolkDevice(device->device);
    {
        std::lock_guard lock(mutex);
        if (saveOnDestroy && dirty && !filePath.empty()) {
            const Result result = saveLocked();
            if (!result) {
                spdlog::warn("Failed to save pipeline cache '{}'", filePath.string());
            }
        }
        vkDestroyPipelineCache(device->device, pipelineCache, nullptr);
        pipelineCache = VK_NULL_HANDLE;
    }
}

Result PipelineCacheImpl::initialize(DeviceImpl& owningDevice, const PipelineCacheDesc& desc)
{
    device = &owningDevice;
    saveOnDestroy = desc.saveOnDestroy;
    fileIdentity = owningDevice.pipelineCacheFileIdentity;
    if (desc.filePath != nullptr && desc.filePath[0] != '\0') {
        filePath = desc.filePath;
        filePathString = filePath.string();
        if (!isPipelineCacheFilePath(filePath)) {
            return makeError(Error::InvalidArgument);
        }
    }

    PipelineCacheFileData fileData;
    PipelineCacheFileLoadStatus fileStatus = PipelineCacheFileLoadStatus::NotFound;
    std::string reason;
    if (!filePath.empty()) {
        fileStatus = loadPipelineCacheFile(filePath, fileIdentity, fileData, reason);
    }
    switch (fileStatus) {
    case PipelineCacheFileLoadStatus::NotFound:
        stats.loadStatus = PipelineCacheLoadStatus::NotFound;
        break;
    case PipelineCacheFileLoadStatus::Loaded:
        stats.loadStatus = PipelineCacheLoadStatus::Loaded;
        break;
    case PipelineCacheFileLoadStatus::Invalid:
        stats.loadStatus = PipelineCacheLoadStatus::Invalid;
        break;
    case PipelineCacheFileLoadStatus::Incompatible:
        stats.loadStatus = PipelineCacheLoadStatus::Incompatible;
        break;
    }

    VkPipelineCacheCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .initialDataSize = fileStatus == PipelineCacheFileLoadStatus::Loaded
            ? fileData.backendData.size()
            : 0,
        .pInitialData = fileStatus == PipelineCacheFileLoadStatus::Loaded &&
                !fileData.backendData.empty()
            ? fileData.backendData.data()
            : nullptr,
    };
    VkResult vkResult = vkCreatePipelineCache(
        owningDevice.device,
        &createInfo,
        nullptr,
        &pipelineCache);
    if (vkResult != VK_SUCCESS && createInfo.initialDataSize > 0) {
        stats.loadStatus = PipelineCacheLoadStatus::Incompatible;
        fileData = {};
        createInfo.initialDataSize = 0;
        createInfo.pInitialData = nullptr;
        vkResult = vkCreatePipelineCache(
            owningDevice.device,
            &createInfo,
            nullptr,
            &pipelineCache);
    }
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    if (stats.loadStatus == PipelineCacheLoadStatus::Loaded) {
        storedPsoHashes.insert(fileData.psoHashes.begin(), fileData.psoHashes.end());
        stats.storedPsoCount = storedPsoHashes.size();
        stats.backendDataSize = fileData.backendData.size();
        spdlog::info(
            "Loaded pipeline cache '{}' with {} PSO hashes and {} backend bytes",
            filePath.string(),
            stats.storedPsoCount,
            stats.backendDataSize);
    } else if (!filePath.empty() &&
               stats.loadStatus != PipelineCacheLoadStatus::NotFound) {
        spdlog::warn(
            "Ignored pipeline cache '{}': {}",
            filePath.string(),
            reason.empty() ? "native cache data is incompatible" : reason);
    }
    return {};
}

Result PipelineCacheImpl::saveLocked()
{
    if (device == nullptr || pipelineCache == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }
    if (filePath.empty()) {
        return {};
    }
    if (!dirty) {
        return {};
    }

    activateVolkDevice(device->device);
    size_t byteSize = 0;
    VkResult vkResult = vkGetPipelineCacheData(
        device->device,
        pipelineCache,
        &byteSize,
        nullptr);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    std::vector<uint8_t> backendData(byteSize);
    for (uint32_t attempt = 0; attempt < 3; ++attempt) {
        size_t writtenSize = backendData.size();
        vkResult = vkGetPipelineCacheData(
            device->device,
            pipelineCache,
            &writtenSize,
            backendData.empty() ? nullptr : backendData.data());
        if (vkResult == VK_SUCCESS) {
            backendData.resize(writtenSize);
            break;
        }
        if (vkResult != VK_INCOMPLETE) {
            return resultFromVk(vkResult);
        }

        byteSize = 0;
        vkResult = vkGetPipelineCacheData(
            device->device,
            pipelineCache,
            &byteSize,
            nullptr);
        if (vkResult != VK_SUCCESS) {
            return resultFromVk(vkResult);
        }
        backendData.resize(byteSize);
    }
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    std::vector<uint64_t> hashes;
    hashes.reserve(storedPsoHashes.size() + sessionPsoHashes.size());
    hashes.insert(hashes.end(), storedPsoHashes.begin(), storedPsoHashes.end());
    hashes.insert(hashes.end(), sessionPsoHashes.begin(), sessionPsoHashes.end());
    std::string reason;
    if (!savePipelineCacheFile(filePath, fileIdentity, hashes, backendData, reason)) {
        spdlog::warn("Failed to write pipeline cache '{}': {}", filePath.string(), reason);
        return makeError(Error::Failure);
    }

    storedPsoHashes.insert(sessionPsoHashes.begin(), sessionPsoHashes.end());
    stats.storedPsoCount = storedPsoHashes.size();
    stats.backendDataSize = backendData.size();
    dirty = false;
    spdlog::info(
        "Saved pipeline cache '{}' with {} PSO hashes and {} backend bytes",
        filePath.string(),
        stats.storedPsoCount,
        stats.backendDataSize);
    return {};
}

bool PipelineCacheImpl::recordPsoLocked(uint64_t psoHash)
{
    const bool cacheHit = storedPsoHashes.contains(psoHash) ||
        sessionPsoHashes.contains(psoHash);
    sessionPsoHashes.insert(psoHash);
    stats.sessionPsoCount = sessionPsoHashes.size();
    if (cacheHit) {
        ++stats.hitCount;
    } else {
        ++stats.missCount;
        dirty = true;
    }
    return cacheHit;
}

void DeviceImpl::addQueue(
    VkQueue queue,
    uint32_t familyIndex,
    uint32_t timestampValidBits,
    QueueType type)
{
    auto impl = std::make_unique<QueueImpl>();
    impl->device = this;
    impl->queue = queue;
    impl->familyIndex = familyIndex;
    impl->timestampValidBits = timestampValidBits;
    impl->type = type;
    queues.emplace_back(new Queue(std::move(impl)));
}

std::vector<uint32_t> queueFamiliesForAccess(const DeviceImpl& device, QueueAccessBits access)
{
    std::vector<uint32_t> families;
    const auto appendUnique = [&families](uint32_t family) {
        if (std::find(families.begin(), families.end(), family) == families.end()) {
            families.push_back(family);
        }
    };
    if (hasFlag(access, QueueAccessBits::Graphics) || access == QueueAccessBits::None) {
        appendUnique(device.graphicsFamily);
    }
    if (hasFlag(access, QueueAccessBits::Compute)) {
        appendUnique(device.computeFamily);
    }
    if (hasFlag(access, QueueAccessBits::Copy)) {
        appendUnique(device.capabilities.independentCopyQueue
            ? device.copyFamily
            : device.graphicsFamily);
    }
    return families;
}

BindlessHeapImpl::~BindlessHeapImpl()
{
    destroyHeapBuffer(samplerHeap);
    destroyHeapBuffer(resourceHeap);
}

Result BindlessHeapImpl::initialize(DeviceImpl& owningDevice, const BindlessHeapDesc& heapDesc)
{
    if (!owningDevice.capabilities.bindlessDescriptorHeap) {
        return makeError(Error::Unsupported);
    }
    if (heapDesc.maxSamplers == 0 &&
        heapDesc.maxSampledImages == 0 &&
        heapDesc.maxStorageImages == 0 &&
        heapDesc.maxBuffers == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (heapDesc.maxSampledImages > std::numeric_limits<uint32_t>::max() - heapDesc.maxStorageImages) {
        return makeError(Error::InvalidArgument);
    }

    device = &owningDevice;
    desc = heapDesc;

    VkResult vkResult = heap.initialize(device->physicalDevice, device->device);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    if (desc.maxSamplers > 0) {
        if (heap.setupSamplerHeap(desc.maxSamplers) == 0) {
            return makeError(Error::Unsupported);
        }
        Result result = createHeapBuffer(heap.samplerHeapSize(), heap.samplerHeapAlignment(), samplerHeap);
        if (!result) {
            return result;
        }
    }

    const uint32_t maxImages = desc.maxSampledImages + desc.maxStorageImages;
    if (maxImages > 0 || desc.maxBuffers > 0) {
        if (heap.setupResourceHeap(maxImages, desc.maxBuffers) == 0) {
            return makeError(Error::Unsupported);
        }
        return createHeapBuffer(heap.resourceHeapSize(), heap.resourceHeapAlignment(), resourceHeap);
    }
    return {};
}

Result BindlessHeapImpl::createHeapBuffer(VkDeviceSize size, VkDeviceSize alignment, BindlessHeapBuffer& outBuffer)
{
    if (device == nullptr || size == 0) {
        return makeError(Error::InvalidArgument);
    }

    const VkDeviceSize paddedSize = size + std::max<VkDeviceSize>(alignment, 1) - 1;
    VkBufferUsageFlags2CreateInfo usage2{
        .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
        .usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT,
    };
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &usage2,
        .size = paddedSize,
        .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo allocationInfo{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    VmaAllocationInfo allocatedInfo{};
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkResult vkResult = vmaCreateBuffer(
        device->allocator,
        &bufferInfo,
        &allocationInfo,
        &buffer,
        &allocation,
        &allocatedInfo);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer,
    };
    const VkDeviceAddress rawAddress = vkGetBufferDeviceAddress(device->device, &addressInfo);
    const VkDeviceAddress address = alignUp(rawAddress, alignment);
    const VkDeviceSize mappedOffset = static_cast<VkDeviceSize>(address - rawAddress);
    if (rawAddress == 0 || address == 0 || mappedOffset + size > paddedSize || allocatedInfo.pMappedData == nullptr) {
        vmaDestroyBuffer(device->allocator, buffer, allocation);
        return makeError(Error::Failure);
    }

    outBuffer = {
        .buffer = buffer,
        .allocation = allocation,
        .mapped = static_cast<uint8_t*>(allocatedInfo.pMappedData) + mappedOffset,
        .address = address,
        .size = size,
        .mappedOffset = mappedOffset,
    };
    return {};
}

void BindlessHeapImpl::destroyHeapBuffer(BindlessHeapBuffer& buffer)
{
    if (device != nullptr && buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(device->allocator, buffer.buffer, buffer.allocation);
    }
    buffer = {};
}

void BindlessHeapImpl::flushSamplerDirty()
{
    if (device == nullptr || samplerHeap.allocation == VK_NULL_HANDLE) {
        return;
    }
    const DescriptorHeap::DirtyRange dirty = heap.samplerDirtyRange();
    if (dirty.size > 0) {
        vmaFlushAllocation(device->allocator, samplerHeap.allocation, samplerHeap.mappedOffset + dirty.offset, dirty.size);
        heap.clearSamplerDirty();
    }
}

void BindlessHeapImpl::flushResourceDirty()
{
    if (device == nullptr || resourceHeap.allocation == VK_NULL_HANDLE) {
        return;
    }

    const DescriptorHeap::DirtyRange dirtyImages = heap.resourceImageDirtyRange();
    if (dirtyImages.size > 0) {
        vmaFlushAllocation(
            device->allocator,
            resourceHeap.allocation,
            resourceHeap.mappedOffset + dirtyImages.offset,
            dirtyImages.size);
    }
    const DescriptorHeap::DirtyRange dirtyBuffers = heap.resourceBufferDirtyRange();
    if (dirtyBuffers.size > 0) {
        vmaFlushAllocation(
            device->allocator,
            resourceHeap.allocation,
            resourceHeap.mappedOffset + dirtyBuffers.offset,
            dirtyBuffers.size);
    }
    heap.clearResourceDirty();
}

SwapchainImpl::~SwapchainImpl()
{
    textures.clear();

    if (swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device->device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    if (surface != VK_NULL_HANDLE) {
        SDL_Vulkan_DestroySurface(device->instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }
}

void SwapchainImpl::wrapImages(const std::vector<VkImage>& images, TextureUsageBits usage)
{
    textures.clear();
    textures.reserve(images.size());

    for (VkImage image : images) {
        TextureDesc textureDesc;
        textureDesc.type = TextureType::Texture2D;
        textureDesc.usage = usage;
        textureDesc.format = format;
        textureDesc.width = width;
        textureDesc.height = height;
        textureDesc.depth = 1;
        textureDesc.mipCount = 1;
        textureDesc.layerCount = 1;

        auto textureImpl = std::make_unique<TextureImpl>();
        textureImpl->device = device;
        textureImpl->desc = textureDesc;
        textureImpl->image = image;
        textureImpl->usage = toVkImageUsage(textureDesc.usage);
        textureImpl->ownsImage = false;
        textures.emplace_back(new Texture(std::move(textureImpl)));
    }
}

Result SwapchainImpl::initialize(const SwapchainDesc& desc)
{
    if (desc.window.system != WindowSystem::Sdl3 || desc.window.nativeWindow == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    auto* window = static_cast<SDL_Window*>(desc.window.nativeWindow);
    if (!SDL_Vulkan_CreateSurface(window, device->instance, nullptr, &surface)) {
        spdlog::error("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
        return makeError(Error::Failure);
    }

    VkBool32 presentSupported = VK_FALSE;
    VkResult vkResult = vkGetPhysicalDeviceSurfaceSupportKHR(
        device->physicalDevice,
        device->graphicsFamily,
        surface,
        &presentSupported);
    if (vkResult != VK_SUCCESS || presentSupported == VK_FALSE) {
        if (vkResult == VK_SUCCESS) {
            return makeError(Error::Unsupported);
        }
        return resultFromVk(vkResult);
    }

    VkSurfaceCapabilitiesKHR capabilities{};
    vkResult = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device->physicalDevice, surface, &capabilities);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    uint32_t surfaceFormatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, surface, &surfaceFormatCount, nullptr);
    if (surfaceFormatCount == 0) {
        return makeError(Error::Unsupported);
    }
    std::vector<VkSurfaceFormatKHR> surfaceFormats(surfaceFormatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        device->physicalDevice,
        surface,
        &surfaceFormatCount,
        surfaceFormats.data());

    const VkFormat requestedFormat = toVkFormat(desc.format);
    VkSurfaceFormatKHR selectedFormat = surfaceFormats.front();
    for (const VkSurfaceFormatKHR& candidate : surfaceFormats) {
        if (candidate.format == requestedFormat &&
            candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            selectedFormat = candidate;
            break;
        }
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device->physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    if (presentModeCount > 0) {
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            device->physicalDevice,
            surface,
            &presentModeCount,
            presentModes.data());
    }

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!desc.vsync) {
        if (std::find(presentModes.begin(), presentModes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != presentModes.end()) {
            presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        } else if (std::find(presentModes.begin(), presentModes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != presentModes.end()) {
            presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
    }

    VkExtent2D extent{};
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = capabilities.currentExtent;
    } else {
        extent.width = std::clamp(
            desc.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        extent.height = std::clamp(
            desc.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);
    }

    uint32_t imageCount = std::max(desc.imageCount, capabilities.minImageCount);
    if (capabilities.maxImageCount != 0) {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }

    VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    TextureUsageBits textureUsage = TextureUsageBits::Present | TextureUsageBits::ColorAttachment;
    if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0) {
        imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        textureUsage = textureUsage | TextureUsageBits::TransferDestination;
    }

    VkSwapchainCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = imageCount,
        .imageFormat = selectedFormat.format,
        .imageColorSpace = selectedFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = imageUsage,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
    };

    vkResult = vkCreateSwapchainKHR(device->device, &createInfo, nullptr, &swapchain);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    uint32_t actualImageCount = 0;
    vkGetSwapchainImagesKHR(device->device, swapchain, &actualImageCount, nullptr);
    std::vector<VkImage> images(actualImageCount);
    vkGetSwapchainImagesKHR(device->device, swapchain, &actualImageCount, images.data());

    vkFormat = selectedFormat.format;
    format = fromVkFormat(selectedFormat.format);
    width = extent.width;
    height = extent.height;
    wrapImages(images, textureUsage);
    return {};
}

} // namespace detail

Queue::Queue(std::unique_ptr<detail::QueueImpl> impl)
    : impl_(std::move(impl))
{
}

Queue::~Queue() = default;
Queue::Queue(Queue&&) noexcept = default;
Queue& Queue::operator=(Queue&&) noexcept = default;

Result Queue::submit(const QueueSubmitDesc& desc)
{
    if (impl_ == nullptr || impl_->queue == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }
    if ((desc.waitSemaphoreCount > 0 && desc.waitSemaphores == nullptr) ||
        (desc.waitSwapchainSemaphoreCount > 0 && desc.waitSwapchainSemaphores == nullptr) ||
        (desc.commandBufferCount > 0 && desc.commandBuffers == nullptr) ||
        (desc.signalSemaphoreCount > 0 && desc.signalSemaphores == nullptr) ||
        (desc.signalSwapchainSemaphoreCount > 0 && desc.signalSwapchainSemaphores == nullptr)) {
        return makeError(Error::InvalidArgument);
    }

    const profiling::NsightProfileRange submitMarker(
        profiling::NsightDomain::Render,
        "Submit",
        profiling::NsightCategory::QueueSubmit,
        desc.commandBufferCount);

    std::vector<VkSemaphoreSubmitInfo> waitSemaphores;
    waitSemaphores.reserve(desc.waitSemaphoreCount + desc.waitSwapchainSemaphoreCount);
    for (uint32_t index = 0; index < desc.waitSemaphoreCount; ++index) {
        const SemaphoreSubmitDesc& wait = desc.waitSemaphores[index];
        if (wait.semaphore == nullptr || wait.semaphore->impl_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        waitSemaphores.push_back({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = wait.semaphore->impl_->semaphore,
            .value = wait.value,
            .stageMask = toVkPipelineStages(wait.stages),
        });
    }
    for (uint32_t index = 0; index < desc.waitSwapchainSemaphoreCount; ++index) {
        const SwapchainSemaphoreSubmitDesc& wait = desc.waitSwapchainSemaphores[index];
        if (wait.semaphore == nullptr || wait.semaphore->impl_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        waitSemaphores.push_back({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = wait.semaphore->impl_->semaphore,
            .stageMask = toVkPipelineStages(wait.stages),
        });
    }

    std::vector<VkCommandBufferSubmitInfo> commandBuffers;
    commandBuffers.reserve(desc.commandBufferCount);
    for (uint32_t index = 0; index < desc.commandBufferCount; ++index) {
        CommandBuffer* commandBuffer = desc.commandBuffers[index];
        if (commandBuffer == nullptr || commandBuffer->impl_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        commandBuffers.push_back({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = commandBuffer->impl_->commandBuffer,
        });
    }

    std::vector<VkSemaphoreSubmitInfo> signalSemaphores;
    signalSemaphores.reserve(desc.signalSemaphoreCount + desc.signalSwapchainSemaphoreCount);
    for (uint32_t index = 0; index < desc.signalSemaphoreCount; ++index) {
        const SemaphoreSubmitDesc& signal = desc.signalSemaphores[index];
        if (signal.semaphore == nullptr || signal.semaphore->impl_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        signalSemaphores.push_back({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = signal.semaphore->impl_->semaphore,
            .value = signal.value,
            .stageMask = toVkPipelineStages(signal.stages),
        });
    }
    for (uint32_t index = 0; index < desc.signalSwapchainSemaphoreCount; ++index) {
        const SwapchainSemaphoreSubmitDesc& signal = desc.signalSwapchainSemaphores[index];
        if (signal.semaphore == nullptr || signal.semaphore->impl_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        signalSemaphores.push_back({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = signal.semaphore->impl_->semaphore,
            .stageMask = toVkPipelineStages(signal.stages),
        });
    }

    VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphoreInfos = waitSemaphores.data(),
        .commandBufferInfoCount = static_cast<uint32_t>(commandBuffers.size()),
        .pCommandBufferInfos = commandBuffers.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphores.size()),
        .pSignalSemaphoreInfos = signalSemaphores.data(),
    };

    VkFence fence = VK_NULL_HANDLE;
    if (desc.signalFence != nullptr) {
        if (desc.signalFence->impl_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        fence = desc.signalFence->impl_->fence;
    }

    return resultFromVk(vkQueueSubmit2(impl_->queue, 1, &submitInfo, fence));
}

Result Queue::waitIdle()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromVk(vkQueueWaitIdle(impl_->queue));
}

QueueType Queue::type() const
{
    return impl_ != nullptr ? impl_->type : QueueType::Graphics;
}

uint32_t Queue::timestampValidBits() const
{
    return impl_ != nullptr ? impl_->timestampValidBits : 0;
}

Fence::Fence(std::unique_ptr<detail::FenceImpl> impl)
    : impl_(std::move(impl))
{
}

Fence::~Fence()
{
    if (impl_ != nullptr && impl_->fence != VK_NULL_HANDLE) {
        vkDestroyFence(impl_->device->device, impl_->fence, nullptr);
        impl_->fence = VK_NULL_HANDLE;
    }
}

Fence::Fence(Fence&&) noexcept = default;
Fence& Fence::operator=(Fence&&) noexcept = default;

Result Fence::wait(uint64_t timeoutNanoseconds)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    const profiling::NsightProfileRange waitMarker(
        profiling::NsightDomain::Render,
        "Fence Wait",
        profiling::NsightCategory::FenceWait,
        timeoutNanoseconds);

    const VkResult result = vkWaitForFences(
        impl_->device->device,
        1,
        &impl_->fence,
        VK_TRUE,
        timeoutNanoseconds);
    if (result == VK_TIMEOUT) {
        return makeError(Error::Failure);
    }
    return resultFromVk(result);
}

Result Fence::reset()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromVk(vkResetFences(impl_->device->device, 1, &impl_->fence));
}

bool Fence::isSignaled() const
{
    return impl_ != nullptr &&
        vkGetFenceStatus(impl_->device->device, impl_->fence) == VK_SUCCESS;
}

TimestampQueryPool::TimestampQueryPool(std::unique_ptr<detail::TimestampQueryPoolImpl> impl)
    : impl_(std::move(impl))
{
}

TimestampQueryPool::~TimestampQueryPool()
{
    if (impl_ != nullptr && impl_->queryPool != VK_NULL_HANDLE) {
        activateVolkDevice(impl_->device->device);
        vkDestroyQueryPool(impl_->device->device, impl_->queryPool, nullptr);
        impl_->queryPool = VK_NULL_HANDLE;
    }
}

TimestampQueryPool::TimestampQueryPool(TimestampQueryPool&&) noexcept = default;
TimestampQueryPool& TimestampQueryPool::operator=(TimestampQueryPool&&) noexcept = default;

const TimestampQueryPoolDesc& TimestampQueryPool::desc() const
{
    static const TimestampQueryPoolDesc kEmptyDesc;
    return impl_ != nullptr ? impl_->desc : kEmptyDesc;
}

Result TimestampQueryPool::readResults(
    uint32_t firstQuery,
    uint32_t queryCount,
    TimestampQueryResult* outResults) const
{
    if (impl_ == nullptr ||
        outResults == nullptr ||
        queryCount == 0 ||
        firstQuery >= impl_->desc.queryCount ||
        queryCount > impl_->desc.queryCount - firstQuery) {
        return makeError(Error::InvalidArgument);
    }

    struct RawTimestampQueryResult {
        uint64_t value = 0;
        uint64_t available = 0;
    };
    std::vector<RawTimestampQueryResult> rawResults(queryCount);
    activateVolkDevice(impl_->device->device);
    const VkResult result = vkGetQueryPoolResults(
        impl_->device->device,
        impl_->queryPool,
        firstQuery,
        queryCount,
        rawResults.size() * sizeof(RawTimestampQueryResult),
        rawResults.data(),
        sizeof(RawTimestampQueryResult),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS && result != VK_NOT_READY) {
        return resultFromVk(result);
    }

    for (uint32_t index = 0; index < queryCount; ++index) {
        outResults[index] = TimestampQueryResult{
            .value = rawResults[index].value,
            .available = rawResults[index].available != 0,
        };
    }
    return {};
}

double TimestampQueryPool::durationMilliseconds(
    uint64_t beginTimestamp,
    uint64_t endTimestamp) const
{
    if (impl_ == nullptr ||
        impl_->timestampValidBits == 0 ||
        impl_->timestampPeriodNanoseconds <= 0.0) {
        return 0.0;
    }

    const uint64_t mask = impl_->timestampValidBits >= 64
        ? std::numeric_limits<uint64_t>::max()
        : (uint64_t{1} << impl_->timestampValidBits) - 1u;
    const uint64_t delta = (endTimestamp - beginTimestamp) & mask;
    return static_cast<double>(delta) * impl_->timestampPeriodNanoseconds / 1'000'000.0;
}

Semaphore::Semaphore(std::unique_ptr<detail::SemaphoreImpl> impl)
    : impl_(std::move(impl))
{
}

Semaphore::~Semaphore()
{
    if (impl_ != nullptr && impl_->semaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(impl_->device->device, impl_->semaphore, nullptr);
        impl_->semaphore = VK_NULL_HANDLE;
    }
}

Semaphore::Semaphore(Semaphore&&) noexcept = default;
Semaphore& Semaphore::operator=(Semaphore&&) noexcept = default;

Result Semaphore::wait(uint64_t value, uint64_t timeoutNanoseconds)
{
    if (impl_ == nullptr || impl_->semaphore == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }

    const profiling::NsightProfileRange waitMarker(
        profiling::NsightDomain::Render,
        "Semaphore Wait",
        profiling::NsightCategory::FenceWait,
        value);

    VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &impl_->semaphore,
        .pValues = &value,
    };
    const VkResult result = vkWaitSemaphores(impl_->device->device, &waitInfo, timeoutNanoseconds);
    if (result == VK_TIMEOUT) {
        return makeError(Error::Failure);
    }
    return resultFromVk(result);
}

Result Semaphore::signal(uint64_t value)
{
    if (impl_ == nullptr || impl_->semaphore == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }

    VkSemaphoreSignalInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = impl_->semaphore,
        .value = value,
    };
    return resultFromVk(vkSignalSemaphore(impl_->device->device, &signalInfo));
}

uint64_t Semaphore::currentValue() const
{
    if (impl_ == nullptr || impl_->semaphore == VK_NULL_HANDLE) {
        return 0;
    }

    uint64_t value = 0;
    const VkResult result = vkGetSemaphoreCounterValue(impl_->device->device, impl_->semaphore, &value);
    if (result != VK_SUCCESS) {
        return 0;
    }
    return value;
}

SwapchainSemaphore::SwapchainSemaphore(std::unique_ptr<detail::SwapchainSemaphoreImpl> impl)
    : impl_(std::move(impl))
{
}

SwapchainSemaphore::~SwapchainSemaphore()
{
    if (impl_ != nullptr && impl_->semaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(impl_->device->device, impl_->semaphore, nullptr);
        impl_->semaphore = VK_NULL_HANDLE;
    }
}

SwapchainSemaphore::SwapchainSemaphore(SwapchainSemaphore&&) noexcept = default;
SwapchainSemaphore& SwapchainSemaphore::operator=(SwapchainSemaphore&&) noexcept = default;

Buffer::Buffer(std::unique_ptr<detail::BufferImpl> impl)
    : impl_(std::move(impl))
{
}

Buffer::~Buffer()
{
    if (impl_ != nullptr) {
        if (impl_->mapped != nullptr) {
            vmaUnmapMemory(impl_->device->allocator, impl_->allocation);
            impl_->mapped = nullptr;
        }
        if (impl_->buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(impl_->device->allocator, impl_->buffer, impl_->allocation);
            impl_->buffer = VK_NULL_HANDLE;
            impl_->allocation = VK_NULL_HANDLE;
        }
    }
}

Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;

const BufferDesc& Buffer::desc() const
{
    static const BufferDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

uint64_t Buffer::deviceAddress() const
{
    if (impl_ == nullptr || impl_->device == nullptr || impl_->buffer == VK_NULL_HANDLE) {
        return 0;
    }
    VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = impl_->buffer,
    };
    return vkGetBufferDeviceAddress(impl_->device->device, &addressInfo);
}

RayTracingAccelerationStructure::RayTracingAccelerationStructure(
    std::unique_ptr<detail::RayTracingAccelerationStructureImpl> impl)
    : impl_(std::move(impl))
{
}

RayTracingAccelerationStructure::~RayTracingAccelerationStructure() = default;
RayTracingAccelerationStructure::RayTracingAccelerationStructure(
    RayTracingAccelerationStructure&&) noexcept = default;
RayTracingAccelerationStructure& RayTracingAccelerationStructure::operator=(
    RayTracingAccelerationStructure&&) noexcept = default;

const RayTracingAccelerationStructureDesc& RayTracingAccelerationStructure::desc() const
{
    static const RayTracingAccelerationStructureDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

bool RayTracingAccelerationStructure::valid() const
{
    return impl_ != nullptr &&
        impl_->accelerationStructure != VK_NULL_HANDLE &&
        impl_->address != 0;
}

PartitionedAccelerationStructure::PartitionedAccelerationStructure(
    std::unique_ptr<detail::PartitionedAccelerationStructureImpl> impl)
    : impl_(std::move(impl))
{
}

PartitionedAccelerationStructure::~PartitionedAccelerationStructure() = default;
PartitionedAccelerationStructure::PartitionedAccelerationStructure(
    PartitionedAccelerationStructure&&) noexcept = default;
PartitionedAccelerationStructure& PartitionedAccelerationStructure::operator=(
    PartitionedAccelerationStructure&&) noexcept = default;

const PartitionedAccelerationStructureDesc& PartitionedAccelerationStructure::desc() const
{
    static const PartitionedAccelerationStructureDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

bool PartitionedAccelerationStructure::valid() const
{
    return impl_ != nullptr && impl_->storage != nullptr && impl_->address != 0;
}

void* Buffer::map()
{
    if (impl_ == nullptr || impl_->allocation == VK_NULL_HANDLE) {
        return nullptr;
    }

    if (impl_->mapped != nullptr) {
        return impl_->mapped;
    }

    if (vmaMapMemory(impl_->device->allocator, impl_->allocation, &impl_->mapped) != VK_SUCCESS) {
        impl_->mapped = nullptr;
    }
    return impl_->mapped;
}

void Buffer::unmap()
{
    if (impl_ != nullptr && impl_->mapped != nullptr) {
        vmaUnmapMemory(impl_->device->allocator, impl_->allocation);
        impl_->mapped = nullptr;
    }
}

void Buffer::flush(uint64_t offset, uint64_t size)
{
    if (impl_ == nullptr || impl_->allocation == VK_NULL_HANDLE) {
        return;
    }

    const VkDeviceSize vkSize = size == UINT64_MAX ? VK_WHOLE_SIZE : size;
    vmaFlushAllocation(impl_->device->allocator, impl_->allocation, offset, vkSize);
}

void Buffer::invalidate(uint64_t offset, uint64_t size)
{
    if (impl_ == nullptr || impl_->allocation == VK_NULL_HANDLE) {
        return;
    }

    const VkDeviceSize vkSize = size == UINT64_MAX ? VK_WHOLE_SIZE : size;
    vmaInvalidateAllocation(impl_->device->allocator, impl_->allocation, offset, vkSize);
}

BufferView::BufferView(std::unique_ptr<detail::BufferViewImpl> impl)
    : impl_(std::move(impl))
{
}

BufferView::~BufferView() = default;
BufferView::BufferView(BufferView&&) noexcept = default;
BufferView& BufferView::operator=(BufferView&&) noexcept = default;

const BufferViewDesc& BufferView::desc() const
{
    static const BufferViewDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

Texture::Texture(std::unique_ptr<detail::TextureImpl> impl)
    : impl_(std::move(impl))
{
}

Texture::~Texture()
{
    if (impl_ != nullptr && impl_->ownsImage && impl_->image != VK_NULL_HANDLE) {
        vmaDestroyImage(impl_->device->allocator, impl_->image, impl_->allocation);
        impl_->image = VK_NULL_HANDLE;
        impl_->allocation = VK_NULL_HANDLE;
    }
}

Texture::Texture(Texture&&) noexcept = default;
Texture& Texture::operator=(Texture&&) noexcept = default;

const TextureDesc& Texture::desc() const
{
    static const TextureDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

TextureView::TextureView(std::unique_ptr<detail::TextureViewImpl> impl)
    : impl_(std::move(impl))
{
}

TextureView::~TextureView()
{
    if (impl_ != nullptr && impl_->view != VK_NULL_HANDLE) {
        vkDestroyImageView(impl_->device->device, impl_->view, nullptr);
        impl_->view = VK_NULL_HANDLE;
    }
}

TextureView::TextureView(TextureView&&) noexcept = default;
TextureView& TextureView::operator=(TextureView&&) noexcept = default;

ShaderModule::ShaderModule(std::unique_ptr<detail::ShaderModuleImpl> impl)
    : impl_(std::move(impl))
{
}

ShaderModule::~ShaderModule()
{
    if (impl_ != nullptr && impl_->module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(impl_->device->device, impl_->module, nullptr);
        impl_->module = VK_NULL_HANDLE;
    }
}

ShaderModule::ShaderModule(ShaderModule&&) noexcept = default;
ShaderModule& ShaderModule::operator=(ShaderModule&&) noexcept = default;

uint64_t ShaderModule::contentHash() const
{
    return impl_ != nullptr ? impl_->contentHash : 0;
}

PipelineCache::PipelineCache(std::unique_ptr<detail::PipelineCacheImpl> impl)
    : impl_(std::move(impl))
{
}

PipelineCache::~PipelineCache() = default;
PipelineCache::PipelineCache(PipelineCache&&) noexcept = default;
PipelineCache& PipelineCache::operator=(PipelineCache&&) noexcept = default;

const char* PipelineCache::filePath() const
{
    return impl_ != nullptr ? impl_->filePathString.c_str() : "";
}

PipelineCacheStats PipelineCache::stats() const
{
    if (impl_ == nullptr) {
        return {};
    }
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}

Result PipelineCache::save()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    std::lock_guard lock(impl_->mutex);
    return impl_->saveLocked();
}

GraphicsPipeline::GraphicsPipeline(std::unique_ptr<detail::GraphicsPipelineImpl> impl)
    : impl_(std::move(impl))
{
}

GraphicsPipeline::~GraphicsPipeline()
{
    if (impl_ != nullptr) {
        if (impl_->pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(impl_->device->device, impl_->pipeline, nullptr);
            impl_->pipeline = VK_NULL_HANDLE;
        }
        if (impl_->layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(impl_->device->device, impl_->layout, nullptr);
            impl_->layout = VK_NULL_HANDLE;
        }
    }
}

GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&&) noexcept = default;
GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&&) noexcept = default;

uint64_t GraphicsPipeline::psoHash() const
{
    return impl_ != nullptr ? impl_->psoHash : 0;
}

bool GraphicsPipeline::pipelineCacheHit() const
{
    return impl_ != nullptr && impl_->pipelineCacheHit;
}

ComputePipeline::ComputePipeline(std::unique_ptr<detail::ComputePipelineImpl> impl)
    : impl_(std::move(impl))
{
}

ComputePipeline::~ComputePipeline()
{
    if (impl_ != nullptr) {
        if (impl_->pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(impl_->device->device, impl_->pipeline, nullptr);
            impl_->pipeline = VK_NULL_HANDLE;
        }
        if (impl_->layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(impl_->device->device, impl_->layout, nullptr);
            impl_->layout = VK_NULL_HANDLE;
        }
    }
}

ComputePipeline::ComputePipeline(ComputePipeline&&) noexcept = default;
ComputePipeline& ComputePipeline::operator=(ComputePipeline&&) noexcept = default;

uint64_t ComputePipeline::psoHash() const
{
    return impl_ != nullptr ? impl_->psoHash : 0;
}

bool ComputePipeline::pipelineCacheHit() const
{
    return impl_ != nullptr && impl_->pipelineCacheHit;
}

GraphicsShaderObjectProgram::GraphicsShaderObjectProgram(
    std::unique_ptr<detail::GraphicsShaderObjectProgramImpl> impl)
    : impl_(std::move(impl))
{
}

GraphicsShaderObjectProgram::~GraphicsShaderObjectProgram()
{
    if (impl_ != nullptr) {
        if (impl_->vertexShader != VK_NULL_HANDLE) {
            vkDestroyShaderEXT(impl_->device->device, impl_->vertexShader, nullptr);
            impl_->vertexShader = VK_NULL_HANDLE;
        }
        if (impl_->fragmentShader != VK_NULL_HANDLE) {
            vkDestroyShaderEXT(impl_->device->device, impl_->fragmentShader, nullptr);
            impl_->fragmentShader = VK_NULL_HANDLE;
        }
    }
}

GraphicsShaderObjectProgram::GraphicsShaderObjectProgram(GraphicsShaderObjectProgram&&) noexcept = default;
GraphicsShaderObjectProgram& GraphicsShaderObjectProgram::operator=(GraphicsShaderObjectProgram&&) noexcept = default;

BindlessHeap::BindlessHeap(std::unique_ptr<detail::BindlessHeapImpl> impl)
    : impl_(std::move(impl))
{
}

BindlessHeap::~BindlessHeap() = default;
BindlessHeap::BindlessHeap(BindlessHeap&&) noexcept = default;
BindlessHeap& BindlessHeap::operator=(BindlessHeap&&) noexcept = default;

const BindlessHeapDesc& BindlessHeap::desc() const
{
    static const BindlessHeapDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

uint32_t BindlessHeap::imageShaderIndexBase() const
{
    return impl_ != nullptr ? impl_->heap.imageShaderIndexBase() : 0;
}

uint32_t BindlessHeap::bufferShaderIndexBase() const
{
    return impl_ != nullptr ? impl_->heap.bufferShaderIndexBase() : 0;
}

Result BindlessHeap::allocateSampler(BindlessHandle& outHandle)
{
    outHandle = {};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->heap.allocateSampler(outHandle)) {
        return makeError(Error::OutOfMemory);
    }
    return {};
}

Result BindlessHeap::allocateSampledImage(BindlessHandle& outHandle)
{
    outHandle = {};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->heap.allocateSampledImage(outHandle)) {
        return makeError(Error::OutOfMemory);
    }
    return {};
}

Result BindlessHeap::allocateStorageImage(BindlessHandle& outHandle)
{
    outHandle = {};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->heap.allocateStorageImage(outHandle)) {
        return makeError(Error::OutOfMemory);
    }
    return {};
}

Result BindlessHeap::allocateBuffer(BindlessHandle& outHandle)
{
    outHandle = {};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->heap.allocateBuffer(outHandle)) {
        return makeError(Error::OutOfMemory);
    }
    return {};
}

void BindlessHeap::release(BindlessHandle handle)
{
    if (impl_ != nullptr) {
        impl_->heap.release(handle);
    }
}

Result BindlessHeap::writeSampler(BindlessHandle handle, const SamplerDesc& sampler)
{
    const BindlessSamplerWrite write{
        .handle = handle,
        .sampler = sampler,
    };
    return writeSamplers(&write, 1);
}

Result BindlessHeap::writeSamplers(const BindlessSamplerWrite* writes, uint32_t writeCount)
{
    if (impl_ == nullptr ||
        impl_->samplerHeap.mapped == nullptr ||
        writes == nullptr ||
        writeCount == 0) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<BindlessHandle> handles(writeCount);
    std::vector<VkSamplerCreateInfo> samplerInfos(writeCount);
    for (uint32_t index = 0; index < writeCount; ++index) {
        const BindlessSamplerWrite& write = writes[index];
        if (write.handle.kind != BindlessHandleKind::Sampler ||
            !std::isfinite(write.sampler.minLod) ||
            !std::isfinite(write.sampler.maxLod) ||
            write.sampler.maxLod < write.sampler.minLod) {
            return makeError(Error::InvalidArgument);
        }
        handles[index] = write.handle;
        samplerInfos[index] = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = toVkSamplerFilter(write.sampler.magFilter),
            .minFilter = toVkSamplerFilter(write.sampler.minFilter),
            .mipmapMode = toVkSamplerMipmapMode(write.sampler.mipFilter),
            .addressModeU = toVkSamplerAddressMode(write.sampler.addressU),
            .addressModeV = toVkSamplerAddressMode(write.sampler.addressV),
            .addressModeW = toVkSamplerAddressMode(write.sampler.addressW),
            .minLod = write.sampler.minLod,
            .maxLod = write.sampler.maxLod,
        };
    }

    const VkResult result = impl_->heap.writeSamplerDescriptors(
        handles.data(),
        samplerInfos.data(),
        writeCount,
        impl_->samplerHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushSamplerDirty();
    return {};
}

Result BindlessHeap::allocateAccelerationStructure(BindlessHandle& outHandle)
{
    outHandle = {};
    if (impl_ == nullptr || !impl_->heap.allocateBuffer(outHandle)) {
        return makeError(impl_ == nullptr ? Error::InvalidArgument : Error::OutOfMemory);
    }
    outHandle.kind = BindlessHandleKind::AccelerationStructure;
    return {};
}

Result BindlessHeap::allocatePartitionedAccelerationStructure(BindlessHandle& outHandle)
{
    outHandle = {};
    if (impl_ == nullptr || !impl_->heap.allocateBuffer(outHandle)) {
        return makeError(impl_ == nullptr ? Error::InvalidArgument : Error::OutOfMemory);
    }
    outHandle.kind = BindlessHandleKind::PartitionedAccelerationStructure;
    return {};
}

Result BindlessHeap::writeSampledImage(BindlessHandle handle, TextureView& view, ResourceState state)
{
    const BindlessImageWrite write{
        .handle = handle,
        .view = &view,
        .state = state,
    };
    return writeImages(&write, 1);
}

Result BindlessHeap::writeStorageImage(BindlessHandle handle, TextureView& view)
{
    const BindlessImageWrite write{
        .handle = handle,
        .view = &view,
        .state = ResourceState::General,
    };
    return writeImages(&write, 1);
}

Result BindlessHeap::writeImages(const BindlessImageWrite* writes, uint32_t writeCount)
{
    if (impl_ == nullptr ||
        impl_->resourceHeap.mapped == nullptr ||
        writes == nullptr ||
        writeCount == 0) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<BindlessHandle> handles(writeCount);
    std::vector<VkImageViewCreateInfo> viewInfos(writeCount);
    std::vector<VkImageDescriptorInfoEXT> imageInfos(writeCount);
    std::vector<VkResourceDescriptorInfoEXT> resourceInfos(writeCount);
    for (uint32_t index = 0; index < writeCount; ++index) {
        const BindlessImageWrite& write = writes[index];
        TextureView* view = write.view;
        const bool sampled = write.handle.kind == BindlessHandleKind::SampledImage;
        const bool storage = write.handle.kind == BindlessHandleKind::StorageImage;
        if ((!sampled && !storage) ||
            view == nullptr ||
            view->impl_ == nullptr ||
            view->impl_->texture == nullptr ||
            view->impl_->texture->impl_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const TextureDesc& textureDesc = view->impl_->texture->impl_->desc;
        if ((sampled && !hasFlag(textureDesc.usage, TextureUsageBits::Sampled)) ||
            (storage && !hasFlag(textureDesc.usage, TextureUsageBits::Storage))) {
            return makeError(Error::InvalidArgument);
        }
        const TextureViewDesc& viewDesc = view->impl_->desc;
        handles[index] = write.handle;
        viewInfos[index] = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = view->impl_->texture->impl_->image,
            .viewType = toVkImageViewType(textureDesc.type),
            .format = view->impl_->format,
            .subresourceRange = {
                .aspectMask = aspectForFormat(textureDesc.format),
                .baseMipLevel = viewDesc.baseMip,
                .levelCount = viewDesc.mipCount,
                .baseArrayLayer = viewDesc.baseLayer,
                .layerCount = viewDesc.layerCount,
            },
        };
        imageInfos[index] = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
            .pView = &viewInfos[index],
            .layout = stateInfo(write.state).layout,
        };
        resourceInfos[index] = {
            .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
            .type = sampled ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .data = {.pImage = &imageInfos[index]},
        };
    }

    const VkResult result = impl_->heap.writeImageDescriptors(
        handles.data(),
        resourceInfos.data(),
        writeCount,
        impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

Result BindlessHeap::writeBufferView(BindlessHandle handle, BufferView& view)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr || view.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    const VkResult result = impl_->heap.writeBufferDescriptor(
        handle,
        view.impl_->address,
        view.impl_->size,
        view.impl_->descriptorType,
        impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

Result BindlessHeap::writeConstantBuffer(BindlessHandle handle, Buffer& buffer)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr || buffer.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer.impl_->buffer,
    };
    const VkDeviceAddress address = vkGetBufferDeviceAddress(impl_->device->device, &addressInfo);
    const VkResult result = impl_->heap.writeBufferDescriptor(
        handle,
        address,
        buffer.impl_->desc.size,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

Result BindlessHeap::writeStorageBuffer(BindlessHandle handle, Buffer& buffer)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr || buffer.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer.impl_->buffer,
    };
    const VkDeviceAddress address = vkGetBufferDeviceAddress(impl_->device->device, &addressInfo);
    const VkResult result = impl_->heap.writeBufferDescriptor(
        handle,
        address,
        buffer.impl_->desc.size,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

Result BindlessHeap::writeAccelerationStructure(
    BindlessHandle handle,
    RayTracingAccelerationStructure& accelerationStructure)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr ||
        accelerationStructure.impl_ == nullptr || !accelerationStructure.valid() ||
        accelerationStructure.impl_->device != impl_->device) {
        return makeError(Error::InvalidArgument);
    }

    const VkResult result = impl_->heap.writeAccelerationStructureDescriptor(
        handle,
        accelerationStructure.impl_->address,
        accelerationStructure.impl_->desc.size,
        impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

Result BindlessHeap::writePartitionedAccelerationStructure(
    BindlessHandle handle,
    PartitionedAccelerationStructure& accelerationStructure)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr ||
        accelerationStructure.impl_ == nullptr || !accelerationStructure.valid() ||
        accelerationStructure.impl_->device != impl_->device) {
        return makeError(Error::InvalidArgument);
    }
    const VkResult result = impl_->heap.writePartitionedAccelerationStructureDescriptor(
        handle,
        accelerationStructure.impl_->address,
        accelerationStructure.impl_->desc.sizes.accelerationStructureSize,
        impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

CommandBuffer::CommandBuffer(std::unique_ptr<detail::CommandBufferImpl> impl)
    : impl_(std::move(impl))
{
}

CommandBuffer::~CommandBuffer()
{
    if (impl_ != nullptr && impl_->commandBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(impl_->device->device, impl_->pool, 1, &impl_->commandBuffer);
        impl_->commandBuffer = VK_NULL_HANDLE;
    }
}

CommandBuffer::CommandBuffer(CommandBuffer&&) noexcept = default;
CommandBuffer& CommandBuffer::operator=(CommandBuffer&&) noexcept = default;

Result CommandBuffer::begin()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    impl_->currentGraphicsPipelineLayout = VK_NULL_HANDLE;
    impl_->currentComputePipelineLayout = VK_NULL_HANDLE;
    impl_->currentGraphicsPipelineBindlessPushStages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    impl_->currentBindlessHeap = nullptr;
    impl_->currentGraphicsPipelineUsesBindlessHeap = false;
    impl_->currentComputePipelineUsesBindlessHeap = false;
    impl_->currentGraphicsShaderObjectUsesBindlessHeap = false;
    impl_->currentGraphicsShaderObjectBound = false;
    impl_->currentBindlessUserDataOffset = sizeof(BindlessHeapPushConstants);
    impl_->hasCurrentViewport = false;
    impl_->hasCurrentScissor = false;
    impl_->currentBindlessUserData.clear();

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    return resultFromVk(vkBeginCommandBuffer(impl_->commandBuffer, &beginInfo));
}

Result CommandBuffer::end()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromVk(vkEndCommandBuffer(impl_->commandBuffer));
}

void CommandBuffer::beginDebugLabel(const DebugLabelDesc& desc)
{
    if (impl_ == nullptr ||
        impl_->device == nullptr ||
        impl_->device->cmdBeginDebugUtilsLabel == nullptr ||
        desc.name == nullptr ||
        desc.name[0] == '\0') {
        return;
    }

    VkDebugUtilsLabelEXT label{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = desc.name,
        .color = {
            desc.color.r,
            desc.color.g,
            desc.color.b,
            desc.color.a,
        },
    };
    impl_->device->cmdBeginDebugUtilsLabel(impl_->commandBuffer, &label);
}

void CommandBuffer::endDebugLabel()
{
    if (impl_ == nullptr ||
        impl_->device == nullptr ||
        impl_->device->cmdEndDebugUtilsLabel == nullptr) {
        return;
    }

    impl_->device->cmdEndDebugUtilsLabel(impl_->commandBuffer);
}

Result CommandBuffer::resetTimestampQueries(
    TimestampQueryPool& queryPool,
    uint32_t firstQuery,
    uint32_t queryCount)
{
    if (impl_ == nullptr ||
        queryPool.impl_ == nullptr ||
        impl_->device != queryPool.impl_->device ||
        queryPool.impl_->queueFamilyIndex != impl_->queueFamilyIndex ||
        queryCount == 0 ||
        firstQuery >= queryPool.impl_->desc.queryCount ||
        queryCount > queryPool.impl_->desc.queryCount - firstQuery) {
        return makeError(Error::InvalidArgument);
    }

    vkCmdResetQueryPool(
        impl_->commandBuffer,
        queryPool.impl_->queryPool,
        firstQuery,
        queryCount);
    return {};
}

Result CommandBuffer::writeTimestamp(
    TimestampQueryPool& queryPool,
    uint32_t queryIndex,
    PipelineStageBits stage)
{
    if (impl_ == nullptr ||
        queryPool.impl_ == nullptr ||
        impl_->device != queryPool.impl_->device ||
        queryPool.impl_->queueFamilyIndex != impl_->queueFamilyIndex ||
        queryIndex >= queryPool.impl_->desc.queryCount) {
        return makeError(Error::InvalidArgument);
    }

    vkCmdWriteTimestamp2(
        impl_->commandBuffer,
        toVkPipelineStages(stage),
        queryPool.impl_->queryPool,
        queryIndex);
    return {};
}

void CommandBuffer::barrier(const BarrierDesc& desc)
{
    if (impl_ == nullptr) {
        return;
    }

    std::vector<VkImageMemoryBarrier2> imageBarriers;
    imageBarriers.reserve(desc.textureCount);

    for (uint32_t index = 0; index < desc.textureCount && desc.textures != nullptr; ++index) {
        const TextureBarrierDesc& barrier = desc.textures[index];
        if (barrier.texture == nullptr || barrier.texture->impl_ == nullptr) {
            continue;
        }

        const StateInfo before = stateInfo(barrier.before);
        const StateInfo after = stateInfo(barrier.after);
        const TextureDesc& textureDesc = barrier.texture->impl_->desc;

        imageBarriers.push_back({
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = before.stage,
            .srcAccessMask = before.access,
            .dstStageMask = after.stage,
            .dstAccessMask = after.access,
            .oldLayout = before.layout,
            .newLayout = after.layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = barrier.texture->impl_->image,
            .subresourceRange = {
                .aspectMask = aspectForFormat(textureDesc.format),
                .baseMipLevel = barrier.baseMip,
                .levelCount = barrier.mipCount,
                .baseArrayLayer = barrier.baseLayer,
                .layerCount = barrier.layerCount,
            },
        });
    }

    std::vector<VkBufferMemoryBarrier2> bufferBarriers;
    bufferBarriers.reserve(desc.bufferCount);

    for (uint32_t index = 0; index < desc.bufferCount && desc.buffers != nullptr; ++index) {
        const BufferBarrierDesc& barrier = desc.buffers[index];
        if (barrier.buffer == nullptr || barrier.buffer->impl_ == nullptr) {
            continue;
        }

        const StateInfo before = stateInfo(barrier.before);
        const StateInfo after = stateInfo(barrier.after);
        bufferBarriers.push_back({
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = before.stage,
            .srcAccessMask = before.access,
            .dstStageMask = after.stage,
            .dstAccessMask = after.access,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = barrier.buffer->impl_->buffer,
            .offset = barrier.offset,
            .size = barrier.size == UINT64_MAX ? VK_WHOLE_SIZE : barrier.size,
        });
    }

    if (imageBarriers.empty() && bufferBarriers.empty()) {
        return;
    }

    VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &dependencyInfo);
}

void CommandBuffer::copyBuffer(const BufferCopyDesc& desc)
{
    if (impl_ == nullptr ||
        desc.source == nullptr ||
        desc.source->impl_ == nullptr ||
        desc.destination == nullptr ||
        desc.destination->impl_ == nullptr ||
        desc.size == 0) {
        return;
    }

    VkBufferCopy copyRegion{
        .srcOffset = desc.sourceOffset,
        .dstOffset = desc.destinationOffset,
        .size = desc.size,
    };
    vkCmdCopyBuffer(
        impl_->commandBuffer,
        desc.source->impl_->buffer,
        desc.destination->impl_->buffer,
        1,
        &copyRegion);
}

void CommandBuffer::copyTexture(const TextureCopyDesc& desc)
{
    if (impl_ == nullptr ||
        desc.source == nullptr ||
        desc.source->impl_ == nullptr ||
        desc.destination == nullptr ||
        desc.destination->impl_ == nullptr ||
        desc.width == 0 ||
        desc.height == 0 ||
        desc.depth == 0) {
        return;
    }

    const VkImageAspectFlags sourceAspect = aspectForFormat(desc.source->impl_->desc.format);
    const VkImageAspectFlags destinationAspect = aspectForFormat(desc.destination->impl_->desc.format);
    if (sourceAspect != destinationAspect) {
        return;
    }

    VkImageCopy copyRegion{
        .srcSubresource = {
            .aspectMask = sourceAspect,
            .mipLevel = desc.sourceMipLevel,
            .baseArrayLayer = desc.sourceBaseLayer,
            .layerCount = 1,
        },
        .srcOffset = {0, 0, 0},
        .dstSubresource = {
            .aspectMask = destinationAspect,
            .mipLevel = desc.destinationMipLevel,
            .baseArrayLayer = desc.destinationBaseLayer,
            .layerCount = 1,
        },
        .dstOffset = {0, 0, 0},
        .extent = {desc.width, desc.height, desc.depth},
    };

    vkCmdCopyImage(
        impl_->commandBuffer,
        desc.source->impl_->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        desc.destination->impl_->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copyRegion);
}

void CommandBuffer::copyTextureToBuffer(const TextureBufferCopyDesc& desc)
{
    if (impl_ == nullptr ||
        desc.texture == nullptr ||
        desc.texture->impl_ == nullptr ||
        desc.buffer == nullptr ||
        desc.buffer->impl_ == nullptr ||
        desc.width == 0 ||
        desc.height == 0 ||
        desc.depth == 0 ||
        desc.layerCount == 0) {
        return;
    }

    uint32_t bufferRowLength = 0;
    uint32_t bufferImageHeight = 0;
    if (!fillBufferImageLayout(
            desc.texture->impl_->desc.format,
            desc.width,
            desc.height,
            desc.bufferRowPitch,
            desc.bufferSlicePitch,
            bufferRowLength,
            bufferImageHeight)) {
        return;
    }

    VkBufferImageCopy copyRegion{
        .bufferOffset = desc.bufferOffset,
        .bufferRowLength = bufferRowLength,
        .bufferImageHeight = bufferImageHeight,
        .imageSubresource = {
            .aspectMask = aspectForFormat(desc.texture->impl_->desc.format),
            .mipLevel = desc.mipLevel,
            .baseArrayLayer = desc.baseLayer,
            .layerCount = desc.layerCount,
        },
        .imageOffset = {desc.textureOffsetX, desc.textureOffsetY, desc.textureOffsetZ},
        .imageExtent = {desc.width, desc.height, desc.depth},
    };

    vkCmdCopyImageToBuffer(
        impl_->commandBuffer,
        desc.texture->impl_->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        desc.buffer->impl_->buffer,
        1,
        &copyRegion);
}

void CommandBuffer::copyBufferToTexture(const BufferTextureCopyDesc& desc)
{
    if (impl_ == nullptr ||
        desc.buffer == nullptr ||
        desc.buffer->impl_ == nullptr ||
        desc.texture == nullptr ||
        desc.texture->impl_ == nullptr ||
        desc.width == 0 ||
        desc.height == 0 ||
        desc.depth == 0 ||
        desc.layerCount == 0) {
        return;
    }

    uint32_t bufferRowLength = 0;
    uint32_t bufferImageHeight = 0;
    if (!fillBufferImageLayout(
            desc.texture->impl_->desc.format,
            desc.width,
            desc.height,
            desc.bufferRowPitch,
            desc.bufferSlicePitch,
            bufferRowLength,
            bufferImageHeight)) {
        return;
    }

    VkBufferImageCopy copyRegion{
        .bufferOffset = desc.bufferOffset,
        .bufferRowLength = bufferRowLength,
        .bufferImageHeight = bufferImageHeight,
        .imageSubresource = {
            .aspectMask = aspectForFormat(desc.texture->impl_->desc.format),
            .mipLevel = desc.mipLevel,
            .baseArrayLayer = desc.baseLayer,
            .layerCount = desc.layerCount,
        },
        .imageOffset = {desc.textureOffsetX, desc.textureOffsetY, desc.textureOffsetZ},
        .imageExtent = {desc.width, desc.height, desc.depth},
    };

    vkCmdCopyBufferToImage(
        impl_->commandBuffer,
        desc.buffer->impl_->buffer,
        desc.texture->impl_->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copyRegion);
}

void CommandBuffer::hostWriteBarrier()
{
    if (impl_ == nullptr) {
        return;
    }
    const VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
            VK_ACCESS_2_MEMORY_WRITE_BIT,
    };
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &dependency);
}

void CommandBuffer::clearColorTexture(Texture& texture, ResourceState state, const ColorValue& color)
{
    if (impl_ == nullptr || texture.impl_ == nullptr || texture.impl_->image == VK_NULL_HANDLE) {
        return;
    }
    const TextureDesc& desc = texture.impl_->desc;
    if (aspectForFormat(desc.format) != VK_IMAGE_ASPECT_COLOR_BIT) {
        return;
    }

    const VkClearColorValue clearValue{{color.r, color.g, color.b, color.a}};
    const VkImageSubresourceRange range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = desc.mipCount,
        .baseArrayLayer = 0,
        .layerCount = desc.layerCount,
    };
    vkCmdClearColorImage(
        impl_->commandBuffer,
        texture.impl_->image,
        stateInfo(state).layout,
        &clearValue,
        1,
        &range);
}

void CommandBuffer::beginRendering(const RenderingDesc& desc)
{
    if (impl_ == nullptr) {
        return;
    }
    if (desc.colorAttachmentCount > 0 && desc.colorAttachments == nullptr) {
        return;
    }

    std::vector<VkRenderingAttachmentInfo> colorAttachments;
    colorAttachments.reserve(desc.colorAttachmentCount);
    VkRenderingAttachmentInfo depthAttachment{};
    const VkRenderingAttachmentInfo* depthAttachmentPtr = nullptr;

    for (uint32_t index = 0; index < desc.colorAttachmentCount; ++index) {
        const RenderingAttachmentDesc& attachment = desc.colorAttachments[index];
        if (attachment.view == nullptr || attachment.view->impl_ == nullptr) {
            continue;
        }

        const StateInfo state = stateInfo(attachment.state);
        const ColorValue& clear = attachment.clearColor;
        colorAttachments.push_back({
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = attachment.view->impl_->view,
            .imageLayout = state.layout,
            .loadOp = toVkLoadOp(attachment.loadOp),
            .storeOp = toVkStoreOp(attachment.storeOp),
            .clearValue = {
                .color = {{clear.r, clear.g, clear.b, clear.a}},
            },
        });
    }

    if (desc.depthStencilAttachment != nullptr) {
        const RenderingAttachmentDesc& attachment = *desc.depthStencilAttachment;
        if (attachment.view != nullptr && attachment.view->impl_ != nullptr) {
            const StateInfo state = stateInfo(attachment.state);
            depthAttachment = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = attachment.view->impl_->view,
                .imageLayout = state.layout,
                .loadOp = toVkLoadOp(attachment.loadOp),
                .storeOp = toVkStoreOp(attachment.storeOp),
                .clearValue = {
                    .depthStencil = {attachment.clearDepth, attachment.clearStencil},
                },
            };
            depthAttachmentPtr = &depthAttachment;
        }
    }

    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {
            .offset = {desc.renderArea.x, desc.renderArea.y},
            .extent = {desc.renderArea.width, desc.renderArea.height},
        },
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
        .pColorAttachments = colorAttachments.data(),
        .pDepthAttachment = depthAttachmentPtr,
    };
    vkCmdBeginRendering(impl_->commandBuffer, &renderingInfo);
}

void CommandBuffer::clearColorAttachment(uint32_t attachmentIndex, const ColorValue& color, const Rect& rect)
{
    if (impl_ == nullptr) {
        return;
    }

    VkClearAttachment attachment{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .colorAttachment = attachmentIndex,
        .clearValue = {
            .color = {{color.r, color.g, color.b, color.a}},
        },
    };

    VkClearRect clearRect{
        .rect = {
            .offset = {rect.x, rect.y},
            .extent = {rect.width, rect.height},
        },
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    vkCmdClearAttachments(impl_->commandBuffer, 1, &attachment, 1, &clearRect);
}

void CommandBuffer::endRendering()
{
    if (impl_ != nullptr) {
        vkCmdEndRendering(impl_->commandBuffer);
    }
}

void CommandBuffer::setViewport(const Viewport& viewport)
{
    if (impl_ == nullptr) {
        return;
    }

    VkViewport vkViewport{
        .x = viewport.x,
        .y = viewport.y,
        .width = viewport.width,
        .height = viewport.height,
        .minDepth = viewport.minDepth,
        .maxDepth = viewport.maxDepth,
    };
    impl_->currentViewport = viewport;
    impl_->hasCurrentViewport = true;
    vkCmdSetViewport(impl_->commandBuffer, 0, 1, &vkViewport);
    if (impl_->currentGraphicsShaderObjectBound && vkCmdSetViewportWithCountEXT != nullptr) {
        vkCmdSetViewportWithCountEXT(impl_->commandBuffer, 1, &vkViewport);
    }
}

void CommandBuffer::setScissor(const Rect& scissor)
{
    if (impl_ == nullptr) {
        return;
    }

    VkRect2D vkScissor{
        .offset = {scissor.x, scissor.y},
        .extent = {scissor.width, scissor.height},
    };
    impl_->currentScissor = scissor;
    impl_->hasCurrentScissor = true;
    vkCmdSetScissor(impl_->commandBuffer, 0, 1, &vkScissor);
    if (impl_->currentGraphicsShaderObjectBound && vkCmdSetScissorWithCountEXT != nullptr) {
        vkCmdSetScissorWithCountEXT(impl_->commandBuffer, 1, &vkScissor);
    }
}

void CommandBuffer::setDepthStencilState(const DepthStencilState& state)
{
    if (impl_ == nullptr) {
        return;
    }

    if (vkCmdSetDepthTestEnableEXT != nullptr) {
        vkCmdSetDepthTestEnableEXT(
            impl_->commandBuffer,
            state.depthTestEnable ? VK_TRUE : VK_FALSE);
    }
    if (vkCmdSetDepthWriteEnableEXT != nullptr) {
        vkCmdSetDepthWriteEnableEXT(
            impl_->commandBuffer,
            state.depthWriteEnable ? VK_TRUE : VK_FALSE);
    }
    if (vkCmdSetDepthCompareOpEXT != nullptr) {
        vkCmdSetDepthCompareOpEXT(
            impl_->commandBuffer,
            toVkCompareOp(state.depthCompareOp));
    }
}

namespace {

void clearGraphicsShaderObjects(detail::CommandBufferImpl& commandBuffer)
{
    if (!commandBuffer.currentGraphicsShaderObjectBound ||
        commandBuffer.device == nullptr ||
        !commandBuffer.device->shaderObjectEnabled ||
        vkCmdBindShadersEXT == nullptr) {
        commandBuffer.currentGraphicsShaderObjectBound = false;
        commandBuffer.currentGraphicsShaderObjectUsesBindlessHeap = false;
        return;
    }

    const std::array<VkShaderStageFlagBits, 5> stages{
        VK_SHADER_STAGE_VERTEX_BIT,
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        VK_SHADER_STAGE_GEOMETRY_BIT,
        VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    vkCmdBindShadersEXT(
        commandBuffer.commandBuffer,
        static_cast<uint32_t>(stages.size()),
        stages.data(),
        nullptr);
    commandBuffer.currentGraphicsShaderObjectBound = false;
    commandBuffer.currentGraphicsShaderObjectUsesBindlessHeap = false;
}

void pushCurrentBindlessData(detail::CommandBufferImpl& commandBuffer, detail::BindlessHeapImpl& heap)
{
    const BindlessHeapPushConstants push{
        .imageShaderIndexBase = heap.heap.imageShaderIndexBase(),
        .bufferShaderIndexBase = heap.heap.bufferShaderIndexBase(),
    };
    if (commandBuffer.currentGraphicsPipelineUsesBindlessHeap &&
        commandBuffer.currentGraphicsPipelineLayout != VK_NULL_HANDLE) {
        vkCmdPushConstants(
            commandBuffer.commandBuffer,
            commandBuffer.currentGraphicsPipelineLayout,
            commandBuffer.currentGraphicsPipelineBindlessPushStages,
            0,
            sizeof(push),
            &push);
    }
    if (commandBuffer.currentComputePipelineUsesBindlessHeap &&
        commandBuffer.currentComputePipelineLayout != VK_NULL_HANDLE) {
        vkCmdPushConstants(
            commandBuffer.commandBuffer,
            commandBuffer.currentComputePipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(push),
            &push);
    }

    const size_t payloadSize = commandBuffer.currentBindlessUserDataOffset +
        commandBuffer.currentBindlessUserData.size();
    const bool needsDescriptorHeapPush =
        commandBuffer.currentGraphicsPipelineUsesBindlessHeap ||
        commandBuffer.currentComputePipelineUsesBindlessHeap ||
        commandBuffer.currentGraphicsShaderObjectUsesBindlessHeap;
    if (needsDescriptorHeapPush &&
        payloadSize > 0 &&
        commandBuffer.device != nullptr &&
        commandBuffer.device->bindlessDescriptorHeapEnabled &&
        commandBuffer.device->descriptorHeapWriter.maxPushDataSize() >= payloadSize &&
        vkCmdPushDataEXT != nullptr) {
        std::vector<uint8_t> payload(payloadSize, 0);
        if (commandBuffer.currentBindlessUserDataOffset != 0) {
            std::memcpy(payload.data(), &push, sizeof(push));
        }
        if (!commandBuffer.currentBindlessUserData.empty()) {
            std::memcpy(
                payload.data() + commandBuffer.currentBindlessUserDataOffset,
                commandBuffer.currentBindlessUserData.data(),
                commandBuffer.currentBindlessUserData.size());
        }

        const VkPushDataInfoEXT pushInfo{
            .sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
            .offset = 0,
            .data = {
                .address = payload.data(),
                .size = payload.size(),
            },
        };
        vkCmdPushDataEXT(commandBuffer.commandBuffer, &pushInfo);
    }
}

} // namespace

void CommandBuffer::bindGraphicsPipeline(GraphicsPipeline& pipeline)
{
    if (impl_ == nullptr || pipeline.impl_ == nullptr) {
        return;
    }
    clearGraphicsShaderObjects(*impl_);
    vkCmdBindPipeline(impl_->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.impl_->pipeline);
    impl_->currentGraphicsPipelineLayout = pipeline.impl_->layout;
    impl_->currentGraphicsPipelineBindlessPushStages = pipeline.impl_->bindlessPushStages;
    impl_->currentGraphicsPipelineUsesBindlessHeap = pipeline.impl_->usesBindlessHeap;
    impl_->currentBindlessUserDataOffset = sizeof(BindlessHeapPushConstants);
    if (impl_->currentGraphicsPipelineUsesBindlessHeap && impl_->currentBindlessHeap != nullptr) {
        pushCurrentBindlessData(*impl_, *impl_->currentBindlessHeap);
    }
}

void CommandBuffer::bindComputePipeline(ComputePipeline& pipeline)
{
    if (impl_ == nullptr || pipeline.impl_ == nullptr) {
        return;
    }
    vkCmdBindPipeline(impl_->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.impl_->pipeline);
    impl_->currentComputePipelineLayout = pipeline.impl_->layout;
    impl_->currentComputePipelineUsesBindlessHeap = pipeline.impl_->usesBindlessHeap;
    impl_->currentBindlessUserDataOffset = pipeline.impl_->bindlessUserDataOffset;
    if (impl_->currentComputePipelineUsesBindlessHeap && impl_->currentBindlessHeap != nullptr) {
        pushCurrentBindlessData(*impl_, *impl_->currentBindlessHeap);
    }
}

void CommandBuffer::bindComputePipeline(
    ComputePipeline& pipeline,
    const void* bindlessData,
    uint32_t byteSize)
{
    if (impl_ == nullptr ||
        pipeline.impl_ == nullptr ||
        (byteSize > 0 && bindlessData == nullptr)) {
        return;
    }
    impl_->currentBindlessUserData.resize(byteSize);
    if (byteSize > 0) {
        std::memcpy(impl_->currentBindlessUserData.data(), bindlessData, byteSize);
    }
    vkCmdBindPipeline(impl_->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.impl_->pipeline);
    impl_->currentComputePipelineLayout = pipeline.impl_->layout;
    impl_->currentComputePipelineUsesBindlessHeap = pipeline.impl_->usesBindlessHeap;
    impl_->currentBindlessUserDataOffset = pipeline.impl_->bindlessUserDataOffset;
    if (impl_->currentComputePipelineUsesBindlessHeap && impl_->currentBindlessHeap != nullptr) {
        pushCurrentBindlessData(*impl_, *impl_->currentBindlessHeap);
    }
}

void CommandBuffer::setGraphicsShaderObjectState()
{
    if (impl_ == nullptr ||
        impl_->device == nullptr ||
        !impl_->device->shaderObjectEnabled) {
        return;
    }

    if (vkCmdSetVertexInputEXT != nullptr) {
        vkCmdSetVertexInputEXT(impl_->commandBuffer, 0, nullptr, 0, nullptr);
    }
    if (vkCmdSetPrimitiveTopologyEXT != nullptr) {
        vkCmdSetPrimitiveTopologyEXT(impl_->commandBuffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    }
    if (vkCmdSetPrimitiveRestartEnableEXT != nullptr) {
        vkCmdSetPrimitiveRestartEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetRasterizerDiscardEnableEXT != nullptr) {
        vkCmdSetRasterizerDiscardEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetPolygonModeEXT != nullptr) {
        vkCmdSetPolygonModeEXT(impl_->commandBuffer, VK_POLYGON_MODE_FILL);
    }
    if (vkCmdSetCullModeEXT != nullptr) {
        vkCmdSetCullModeEXT(impl_->commandBuffer, VK_CULL_MODE_NONE);
    }
    if (vkCmdSetFrontFaceEXT != nullptr) {
        vkCmdSetFrontFaceEXT(impl_->commandBuffer, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    }
    if (vkCmdSetDepthClampEnableEXT != nullptr) {
        vkCmdSetDepthClampEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetDepthBiasEnableEXT != nullptr) {
        vkCmdSetDepthBiasEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    vkCmdSetLineWidth(impl_->commandBuffer, 1.0f);
    if (vkCmdSetRasterizationSamplesEXT != nullptr) {
        vkCmdSetRasterizationSamplesEXT(impl_->commandBuffer, VK_SAMPLE_COUNT_1_BIT);
    }
    if (vkCmdSetSampleMaskEXT != nullptr) {
        const VkSampleMask sampleMask = 0xffffffffu;
        vkCmdSetSampleMaskEXT(impl_->commandBuffer, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
    }
    if (vkCmdSetAlphaToCoverageEnableEXT != nullptr) {
        vkCmdSetAlphaToCoverageEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetAlphaToOneEnableEXT != nullptr) {
        vkCmdSetAlphaToOneEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetDepthTestEnableEXT != nullptr) {
        vkCmdSetDepthTestEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetDepthWriteEnableEXT != nullptr) {
        vkCmdSetDepthWriteEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetDepthCompareOpEXT != nullptr) {
        vkCmdSetDepthCompareOpEXT(impl_->commandBuffer, VK_COMPARE_OP_ALWAYS);
    }
    if (vkCmdSetDepthBoundsTestEnableEXT != nullptr) {
        vkCmdSetDepthBoundsTestEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetStencilTestEnableEXT != nullptr) {
        vkCmdSetStencilTestEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetColorBlendEnableEXT != nullptr) {
        const VkBool32 blendEnable = VK_FALSE;
        vkCmdSetColorBlendEnableEXT(impl_->commandBuffer, 0, 1, &blendEnable);
    }
    if (vkCmdSetColorBlendEquationEXT != nullptr) {
        const VkColorBlendEquationEXT blendEquation{
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
        };
        vkCmdSetColorBlendEquationEXT(impl_->commandBuffer, 0, 1, &blendEquation);
    }
    if (vkCmdSetColorWriteMaskEXT != nullptr) {
        const VkColorComponentFlags colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        vkCmdSetColorWriteMaskEXT(impl_->commandBuffer, 0, 1, &colorWriteMask);
    }
    if (vkCmdSetLogicOpEnableEXT != nullptr) {
        vkCmdSetLogicOpEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (vkCmdSetLogicOpEXT != nullptr) {
        vkCmdSetLogicOpEXT(impl_->commandBuffer, VK_LOGIC_OP_COPY);
    }

    if (impl_->hasCurrentViewport && vkCmdSetViewportWithCountEXT != nullptr) {
        const VkViewport viewport{
            .x = impl_->currentViewport.x,
            .y = impl_->currentViewport.y,
            .width = impl_->currentViewport.width,
            .height = impl_->currentViewport.height,
            .minDepth = impl_->currentViewport.minDepth,
            .maxDepth = impl_->currentViewport.maxDepth,
        };
        vkCmdSetViewportWithCountEXT(impl_->commandBuffer, 1, &viewport);
    }
    if (impl_->hasCurrentScissor && vkCmdSetScissorWithCountEXT != nullptr) {
        const VkRect2D scissor{
            .offset = {impl_->currentScissor.x, impl_->currentScissor.y},
            .extent = {impl_->currentScissor.width, impl_->currentScissor.height},
        };
        vkCmdSetScissorWithCountEXT(impl_->commandBuffer, 1, &scissor);
    }
}

void CommandBuffer::bindGraphicsShaderObjectProgram(GraphicsShaderObjectProgram& program)
{
    if (impl_ == nullptr ||
        program.impl_ == nullptr ||
        impl_->device == nullptr ||
        !impl_->device->shaderObjectEnabled ||
        vkCmdBindShadersEXT == nullptr) {
        return;
    }

    const std::array<VkShaderStageFlagBits, 5> stages{
        VK_SHADER_STAGE_VERTEX_BIT,
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        VK_SHADER_STAGE_GEOMETRY_BIT,
        VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const std::array<VkShaderEXT, 5> shaders{
        program.impl_->vertexShader,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        program.impl_->fragmentShader,
    };
    vkCmdBindShadersEXT(
        impl_->commandBuffer,
        static_cast<uint32_t>(stages.size()),
        stages.data(),
        shaders.data());

    impl_->currentGraphicsPipelineLayout = VK_NULL_HANDLE;
    impl_->currentGraphicsPipelineUsesBindlessHeap = false;
    impl_->currentGraphicsShaderObjectBound = true;
    impl_->currentGraphicsShaderObjectUsesBindlessHeap = program.impl_->usesBindlessHeap;
    impl_->currentBindlessUserDataOffset = sizeof(BindlessHeapPushConstants);
    if (impl_->currentGraphicsShaderObjectUsesBindlessHeap && impl_->currentBindlessHeap != nullptr) {
        pushCurrentBindlessData(*impl_, *impl_->currentBindlessHeap);
    }
}

void CommandBuffer::bindBindlessHeap(BindlessHeap& heap)
{
    if (impl_ == nullptr || heap.impl_ == nullptr) {
        return;
    }
    if (impl_->currentBindlessHeap == heap.impl_.get()) {
        return;
    }

    impl_->currentBindlessHeap = heap.impl_.get();
    heap.impl_->heap.bind(
        impl_->commandBuffer,
        heap.impl_->samplerHeap.address,
        heap.impl_->resourceHeap.address);
    pushCurrentBindlessData(*impl_, *heap.impl_);
}

void CommandBuffer::pushBindlessData(const void* data, uint32_t byteSize)
{
    if (impl_ == nullptr || (byteSize > 0 && data == nullptr)) {
        return;
    }

    impl_->currentBindlessUserData.resize(byteSize);
    if (byteSize > 0) {
        std::memcpy(impl_->currentBindlessUserData.data(), data, byteSize);
    }
    if (impl_->currentBindlessHeap != nullptr) {
        pushCurrentBindlessData(*impl_, *impl_->currentBindlessHeap);
    }
}

void CommandBuffer::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    if (impl_ != nullptr) {
        vkCmdDraw(impl_->commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
    }
}

void CommandBuffer::drawMeshTasks(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    if (impl_ == nullptr || groupCountX == 0 || groupCountY == 0 || groupCountZ == 0) {
        return;
    }
#ifdef VK_EXT_mesh_shader
    if (vkCmdDrawMeshTasksEXT != nullptr) {
        vkCmdDrawMeshTasksEXT(impl_->commandBuffer, groupCountX, groupCountY, groupCountZ);
    }
#endif
}

void CommandBuffer::drawMeshTasksIndirect(Buffer& buffer, uint64_t offset)
{
    if (impl_ == nullptr ||
        buffer.impl_ == nullptr ||
        !hasFlag(buffer.impl_->desc.usage, BufferUsageBits::Indirect) ||
        (offset & 3u) != 0 ||
        offset > buffer.impl_->desc.size ||
        sizeof(VkDrawMeshTasksIndirectCommandEXT) > buffer.impl_->desc.size - offset) {
        return;
    }
#ifdef VK_EXT_mesh_shader
    if (vkCmdDrawMeshTasksIndirectEXT != nullptr) {
        vkCmdDrawMeshTasksIndirectEXT(impl_->commandBuffer, buffer.impl_->buffer, offset, 1, 0);
    }
#endif
}

void CommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    if (impl_ != nullptr && groupCountX > 0 && groupCountY > 0 && groupCountZ > 0) {
        vkCmdDispatch(impl_->commandBuffer, groupCountX, groupCountY, groupCountZ);
    }
}

Result CommandBuffer::buildRayTracingAccelerationStructure(
    const RayTracingAccelerationStructureBuildDesc& desc)
{
    if (impl_ == nullptr || desc.destination == nullptr ||
        desc.destination->impl_ == nullptr || !desc.destination->valid() ||
        desc.destination->impl_->device != impl_->device ||
        desc.scratchBuffer == nullptr || desc.scratchBuffer->impl_ == nullptr ||
        desc.scratchBuffer->impl_->device != impl_->device ||
        !hasFlag(desc.scratchBuffer->desc().usage, BufferUsageBits::Storage)) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->device->rayTracingAccelerationStructureEnabled ||
        !impl_->device->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    const RayTracingAccelerationStructureDesc& destinationDesc =
        desc.destination->impl_->desc;
    if (desc.mode == RayTracingAccelerationStructureBuildMode::Update) {
        if (!hasFlag(
                destinationDesc.buildFlags,
                RayTracingAccelerationStructureBuildFlags::AllowUpdate) ||
            desc.source == nullptr || desc.source->impl_ == nullptr ||
            !desc.source->valid() || desc.source->impl_->device != impl_->device ||
            desc.source->impl_->desc.type != destinationDesc.type) {
            return makeError(Error::InvalidArgument);
        }
    } else if (desc.source != nullptr) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    if (destinationDesc.type == RayTracingAccelerationStructureType::BottomLevel) {
        if (desc.geometries == nullptr || desc.geometryCount == 0 ||
            desc.instanceBuffer != nullptr || desc.instanceCount != 0) {
            return makeError(Error::InvalidArgument);
        }
        geometries.reserve(desc.geometryCount);
        ranges.reserve(desc.geometryCount);
        for (uint32_t index = 0; index < desc.geometryCount; ++index) {
            const RayTracingTriangleGeometryDesc& source = desc.geometries[index];
            if (source.vertexBuffer == nullptr || source.vertexBuffer->impl_ == nullptr ||
                source.vertexBuffer->impl_->device != impl_->device ||
                source.vertexCount == 0 || source.vertexStride == 0 ||
                source.primitiveCount == 0 || source.vertexFormat == Format::Unknown ||
                source.vertexOffset >= source.vertexBuffer->desc().size ||
                !hasFlag(
                    source.vertexBuffer->desc().usage,
                    BufferUsageBits::AccelerationStructureBuildInput)) {
                return makeError(Error::InvalidArgument);
            }
            const VkDeviceAddress vertexAddress = source.vertexBuffer->deviceAddress();
            const VkFormat vertexFormat = toVkFormat(source.vertexFormat);
            if (vertexAddress == 0 || vertexFormat == VK_FORMAT_UNDEFINED) {
                return makeError(Error::Failure);
            }

            VkDeviceAddress indexAddress = 0;
            if (source.indexType != RayTracingIndexType::None) {
                if (source.indexBuffer == nullptr || source.indexBuffer->impl_ == nullptr ||
                    source.indexBuffer->impl_->device != impl_->device ||
                    source.indexOffset >= source.indexBuffer->desc().size ||
                    !hasFlag(
                        source.indexBuffer->desc().usage,
                        BufferUsageBits::AccelerationStructureBuildInput)) {
                    return makeError(Error::InvalidArgument);
                }
                indexAddress = source.indexBuffer->deviceAddress();
                if (indexAddress == 0) {
                    return makeError(Error::Failure);
                }
            }

            VkAccelerationStructureGeometryTrianglesDataKHR triangles{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .vertexFormat = vertexFormat,
                .vertexData = {.deviceAddress = vertexAddress + source.vertexOffset},
                .vertexStride = source.vertexStride,
                .maxVertex = source.vertexCount - 1,
                .indexType = toVkRayTracingIndexType(source.indexType),
                .indexData = {.deviceAddress = indexAddress == 0 ? 0 : indexAddress + source.indexOffset},
            };
            VkAccelerationStructureGeometryKHR geometry{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                .flags = toVkGeometryFlags(source.flags),
            };
            geometry.geometry.triangles = triangles;
            geometries.push_back(geometry);
            ranges.push_back(VkAccelerationStructureBuildRangeInfoKHR{
                .primitiveCount = source.primitiveCount,
            });
        }
    } else {
        if (desc.geometries != nullptr || desc.geometryCount != 0 ||
            desc.instanceBuffer == nullptr || desc.instanceBuffer->impl_ == nullptr ||
            desc.instanceBuffer->impl_->device != impl_->device || desc.instanceCount == 0 ||
            !hasFlag(
                desc.instanceBuffer->desc().usage,
                BufferUsageBits::AccelerationStructureBuildInput)) {
            return makeError(Error::InvalidArgument);
        }
        const VkDeviceAddress instanceAddress = desc.instanceBuffer->deviceAddress();
        if (instanceAddress == 0 || (instanceAddress & 15u) != 0) {
            return makeError(Error::Failure);
        }
        VkAccelerationStructureGeometryInstancesDataKHR instances{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            .arrayOfPointers = VK_FALSE,
            .data = {.deviceAddress = instanceAddress},
        };
        VkAccelerationStructureGeometryKHR geometry{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        };
        geometry.geometry.instances = instances;
        geometries.push_back(geometry);
        ranges.push_back(VkAccelerationStructureBuildRangeInfoKHR{
            .primitiveCount = desc.instanceCount,
        });
    }

    const VkDeviceAddress scratchBase = desc.scratchBuffer->deviceAddress();
    if (scratchBase == 0 || desc.scratchBufferOffset >= desc.scratchBuffer->desc().size) {
        return makeError(Error::Failure);
    }
    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    };
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &properties,
    };
    vkGetPhysicalDeviceProperties2(impl_->device->physicalDevice, &properties2);
    const uint64_t scratchAlignment = std::max<uint64_t>(
        1,
        properties.minAccelerationStructureScratchOffsetAlignment);
    const VkDeviceAddress unalignedScratchAddress = scratchBase + desc.scratchBufferOffset;
    const VkDeviceAddress scratchAddress =
        (unalignedScratchAddress + scratchAlignment - 1u) & ~(scratchAlignment - 1u);
    const uint64_t alignedScratchOffset = scratchAddress - scratchBase;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = toVkAccelerationStructureType(destinationDesc.type),
        .flags = toVkAccelerationStructureBuildFlags(destinationDesc.buildFlags),
        .mode = desc.mode == RayTracingAccelerationStructureBuildMode::Update
            ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
            : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = desc.source != nullptr
            ? desc.source->impl_->accelerationStructure
            : VK_NULL_HANDLE,
        .dstAccelerationStructure = desc.destination->impl_->accelerationStructure,
        .geometryCount = static_cast<uint32_t>(geometries.size()),
        .pGeometries = geometries.data(),
        .scratchData = {.deviceAddress = scratchAddress},
    };
    std::vector<uint32_t> primitiveCounts;
    primitiveCounts.reserve(ranges.size());
    for (const VkAccelerationStructureBuildRangeInfoKHR& range : ranges) {
        primitiveCounts.push_back(range.primitiveCount);
    }
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetAccelerationStructureBuildSizesKHR(
        impl_->device->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        primitiveCounts.data(),
        &sizes);
    const uint64_t requiredScratchSize =
        desc.mode == RayTracingAccelerationStructureBuildMode::Update
        ? sizes.updateScratchSize
        : sizes.buildScratchSize;
    if (requiredScratchSize == 0 ||
        alignedScratchOffset >= desc.scratchBuffer->desc().size ||
        requiredScratchSize > desc.scratchBuffer->desc().size - alignedScratchOffset ||
        sizes.accelerationStructureSize > destinationDesc.size) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers;
    rangePointers.reserve(ranges.size());
    for (const VkAccelerationStructureBuildRangeInfoKHR& range : ranges) {
        rangePointers.push_back(&range);
    }
    vkCmdBuildAccelerationStructuresKHR(
        impl_->commandBuffer,
        1,
        &buildInfo,
        rangePointers.data());

    const VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            VK_ACCESS_2_SHADER_READ_BIT,
    };
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &dependency);
    return {};
}

Result CommandBuffer::buildClusterAccelerationStructureTriangles(
    const ClusterAccelerationStructureTriangleBuildDesc& desc)
{
#ifndef VK_NV_cluster_acceleration_structure
    (void)desc;
    return makeError(Error::Unsupported);
#else
    if (impl_ == nullptr ||
        impl_->device == nullptr ||
        !impl_->device->clusterAccelerationStructureEnabled ||
        vkCmdBuildClusterAccelerationStructureIndirectNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    if (desc.clusters == nullptr ||
        desc.clusterCount == 0 ||
        desc.maxClusterTriangleCount == 0 ||
        desc.maxClusterVertexCount == 0 ||
        desc.maxClusterUniqueGeometryCount == 0 ||
        desc.vertexFormat == Format::Unknown ||
        desc.scratchBuffer == nullptr ||
        desc.buildInfoBuffer == nullptr ||
        desc.destinationAddressBuffer == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    Buffer* scratchBuffer = desc.scratchBuffer;
    Buffer* buildInfoBuffer = desc.buildInfoBuffer;
    Buffer* destinationAddressBuffer = desc.destinationAddressBuffer;
    if (scratchBuffer->impl_ == nullptr ||
        buildInfoBuffer->impl_ == nullptr ||
        destinationAddressBuffer->impl_ == nullptr ||
        scratchBuffer->impl_->device != impl_->device ||
        buildInfoBuffer->impl_->device != impl_->device ||
        destinationAddressBuffer->impl_->device != impl_->device ||
        !hasFlag(scratchBuffer->desc().usage, BufferUsageBits::ShaderDeviceAddress) ||
        !hasFlag(buildInfoBuffer->desc().usage, BufferUsageBits::AccelerationStructureBuildInput) ||
        !hasFlag(buildInfoBuffer->desc().usage, BufferUsageBits::ShaderDeviceAddress) ||
        !hasFlag(destinationAddressBuffer->desc().usage, BufferUsageBits::AccelerationStructureBuildInput) ||
        !hasFlag(destinationAddressBuffer->desc().usage, BufferUsageBits::ShaderDeviceAddress)) {
        return makeError(Error::InvalidArgument);
    }

    const uint64_t buildInfoBytes = static_cast<uint64_t>(desc.clusterCount) *
        sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV);
    const uint64_t destinationAddressBytes =
        static_cast<uint64_t>(desc.clusterCount) * sizeof(uint64_t);
    if (buildInfoBytes > buildInfoBuffer->desc().size ||
        destinationAddressBytes > destinationAddressBuffer->desc().size ||
        desc.scratchBufferOffset > scratchBuffer->desc().size) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> buildInfos(
        desc.clusterCount);
    std::vector<uint64_t> destinationAddresses(desc.clusterCount);
    std::vector<std::pair<Buffer*, uint64_t>> bufferAddresses;
    bufferAddresses.reserve(6);
    auto bufferAddress = [&bufferAddresses](Buffer& buffer) {
        const auto iter = std::find_if(
            bufferAddresses.begin(),
            bufferAddresses.end(),
            [&buffer](const auto& entry) { return entry.first == &buffer; });
        if (iter != bufferAddresses.end()) {
            return iter->second;
        }
        const uint64_t address = buffer.deviceAddress();
        bufferAddresses.emplace_back(&buffer, address);
        return address;
    };
    uint64_t totalTriangleCount = 0;
    uint64_t totalVertexCount = 0;
    for (uint32_t index = 0; index < desc.clusterCount; ++index) {
        const ClusterAccelerationStructureTriangleBuildInfo& source = desc.clusters[index];
        if (source.triangleCount == 0 ||
            source.vertexCount == 0 ||
            source.triangleCount > desc.maxClusterTriangleCount ||
            source.vertexCount > desc.maxClusterVertexCount ||
            source.triangleCount > 0x1ffu ||
            source.vertexCount > 0x1ffu ||
            source.positionTruncateBitCount > 0x3fu ||
            source.geometryIndex > desc.maxGeometryIndexValue ||
            source.geometryIndex > 0xffffffu ||
            source.indexBufferStride == 0 ||
            source.vertexBufferStride == 0 ||
            source.indexBuffer == nullptr ||
            source.vertexBuffer == nullptr ||
            source.destinationBuffer == nullptr ||
            source.destinationSize == 0) {
            return makeError(Error::InvalidArgument);
        }
        Buffer* indexBuffer = source.indexBuffer;
        Buffer* vertexBuffer = source.vertexBuffer;
        Buffer* destinationBuffer = source.destinationBuffer;
        if (indexBuffer->impl_ == nullptr ||
            vertexBuffer->impl_ == nullptr ||
            destinationBuffer->impl_ == nullptr ||
            indexBuffer->impl_->device != impl_->device ||
            vertexBuffer->impl_->device != impl_->device ||
            destinationBuffer->impl_->device != impl_->device ||
            !hasFlag(indexBuffer->desc().usage, BufferUsageBits::AccelerationStructureBuildInput) ||
            !hasFlag(vertexBuffer->desc().usage, BufferUsageBits::AccelerationStructureBuildInput) ||
            !hasFlag(destinationBuffer->desc().usage, BufferUsageBits::AccelerationStructureStorage)) {
            return makeError(Error::InvalidArgument);
        }

        const uint64_t indexElementSize = clusterIndexByteSize(source.indexFormat);
        const uint64_t indexCount = static_cast<uint64_t>(source.triangleCount) * 3u;
        const uint64_t requiredIndexBytes = indexCount == 0
            ? 0
            : (indexCount - 1u) * source.indexBufferStride + indexElementSize;
        const uint64_t requiredVertexBytes =
            static_cast<uint64_t>(source.vertexCount) * source.vertexBufferStride;
        if (indexElementSize == 0 ||
            source.indexBufferStride < indexElementSize ||
            source.indexBufferOffset > indexBuffer->desc().size ||
            requiredIndexBytes > indexBuffer->desc().size - source.indexBufferOffset ||
            source.vertexBufferOffset > vertexBuffer->desc().size ||
            requiredVertexBytes > vertexBuffer->desc().size - source.vertexBufferOffset ||
            source.destinationBufferOffset > destinationBuffer->desc().size ||
            source.destinationSize >
                destinationBuffer->desc().size - source.destinationBufferOffset) {
            return makeError(Error::InvalidArgument);
        }

        const uint64_t indexAddress = bufferAddress(*indexBuffer);
        const uint64_t vertexAddress = bufferAddress(*vertexBuffer);
        const uint64_t destinationAddress = bufferAddress(*destinationBuffer);
        if (indexAddress == 0 || vertexAddress == 0 || destinationAddress == 0) {
            return makeError(Error::Failure);
        }

        VkClusterAccelerationStructureBuildTriangleClusterInfoNV& buildInfo = buildInfos[index];
        buildInfo.clusterID = source.clusterId;
        buildInfo.triangleCount = source.triangleCount;
        buildInfo.vertexCount = source.vertexCount;
        buildInfo.positionTruncateBitCount = source.positionTruncateBitCount;
        buildInfo.indexType = toVkClusterIndexFormat(source.indexFormat);
        buildInfo.baseGeometryIndexAndGeometryFlags.geometryIndex = source.geometryIndex;
        buildInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = source.opaque
            ? VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV
            : 0;
        buildInfo.indexBufferStride = source.indexBufferStride;
        buildInfo.vertexBufferStride = source.vertexBufferStride;
        buildInfo.indexBuffer = indexAddress + source.indexBufferOffset;
        buildInfo.vertexBuffer = vertexAddress + source.vertexBufferOffset;
        destinationAddresses[index] = destinationAddress + source.destinationBufferOffset;

        totalTriangleCount += source.triangleCount;
        totalVertexCount += source.vertexCount;
        if (totalTriangleCount > std::numeric_limits<uint32_t>::max() ||
            totalVertexCount > std::numeric_limits<uint32_t>::max()) {
            return makeError(Error::InvalidArgument);
        }
    }

    void* mappedBuildInfos = buildInfoBuffer->map();
    void* mappedDestinations = destinationAddressBuffer->map();
    if (mappedBuildInfos == nullptr || mappedDestinations == nullptr) {
        if (mappedBuildInfos != nullptr) {
            buildInfoBuffer->unmap();
        }
        if (mappedDestinations != nullptr) {
            destinationAddressBuffer->unmap();
        }
        return makeError(Error::Failure);
    }
    std::memcpy(mappedBuildInfos, buildInfos.data(), static_cast<size_t>(buildInfoBytes));
    std::memcpy(
        mappedDestinations,
        destinationAddresses.data(),
        static_cast<size_t>(destinationAddressBytes));
    buildInfoBuffer->flush(0, buildInfoBytes);
    destinationAddressBuffer->flush(0, destinationAddressBytes);
    buildInfoBuffer->unmap();
    destinationAddressBuffer->unmap();

    const uint64_t buildInfoAddress = bufferAddress(*buildInfoBuffer);
    const uint64_t destinationAddress = bufferAddress(*destinationAddressBuffer);
    const uint64_t scratchAddress = bufferAddress(*scratchBuffer);
    if (buildInfoAddress == 0 || destinationAddress == 0 || scratchAddress == 0) {
        return makeError(Error::Failure);
    }

    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV,
        .vertexFormat = toVkFormat(desc.vertexFormat),
        .maxGeometryIndexValue = desc.maxGeometryIndexValue,
        .maxClusterUniqueGeometryCount = desc.maxClusterUniqueGeometryCount,
        .maxClusterTriangleCount = desc.maxClusterTriangleCount,
        .maxClusterVertexCount = desc.maxClusterVertexCount,
        .maxTotalTriangleCount = static_cast<uint32_t>(totalTriangleCount),
        .maxTotalVertexCount = static_cast<uint32_t>(totalVertexCount),
        .minPositionTruncateBitCount = desc.minPositionTruncateBitCount,
    };
    VkClusterAccelerationStructureInputInfoNV input{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = desc.clusterCount,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
        .opInput = {.pTriangleClusters = &triangleInput},
    };
    VkClusterAccelerationStructureCommandsInfoNV commands{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
        .input = input,
        .scratchData = scratchAddress + desc.scratchBufferOffset,
        .dstAddressesArray = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = destinationAddress,
            .stride = sizeof(uint64_t),
            .size = destinationAddressBytes,
        },
        .srcInfosArray = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = buildInfoAddress,
            .stride = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
            .size = buildInfoBytes,
        },
    };

    const VkMemoryBarrier2 inputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    const VkDependencyInfo inputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &inputBarrier,
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &inputDependency);
    vkCmdBuildClusterAccelerationStructureIndirectNV(impl_->commandBuffer, &commands);

    const VkMemoryBarrier2 outputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_SHADER_READ_BIT,
    };
    const VkDependencyInfo outputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &outputBarrier,
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &outputDependency);
    return {};
#endif
}

Result CommandBuffer::buildClusterAccelerationStructureBottomLevels(
    const ClusterAccelerationStructureBottomLevelBuildDesc& desc)
{
#ifndef VK_NV_cluster_acceleration_structure
    (void)desc;
    return makeError(Error::Unsupported);
#else
    if (impl_ == nullptr || impl_->device == nullptr ||
        !impl_->device->clusterAccelerationStructureEnabled ||
        vkCmdBuildClusterAccelerationStructureIndirectNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    if (desc.maxClusterCountPerAccelerationStructure == 0 ||
        desc.maxTotalClusterCount == 0 ||
        desc.maxAccelerationStructureCount == 0 ||
        desc.buildInfoBuffer == nullptr ||
        desc.destinationAddressBuffer == nullptr ||
        desc.scratchBuffer == nullptr ||
        desc.buildInfoStride < sizeof(ClusterAccelerationStructureBottomLevelBuildInfo) ||
        desc.destinationAddressStride < sizeof(uint64_t)) {
        return makeError(Error::InvalidArgument);
    }

    auto validateBuffer = [this](Buffer* buffer, BufferUsageBits usage) {
        return buffer != nullptr && buffer->impl_ != nullptr &&
            buffer->impl_->device == impl_->device &&
            hasFlag(buffer->desc().usage, BufferUsageBits::ShaderDeviceAddress) &&
            hasFlag(buffer->desc().usage, usage);
    };
    if (!validateBuffer(
            desc.buildInfoBuffer,
            BufferUsageBits::AccelerationStructureBuildInput) ||
        !validateBuffer(desc.destinationAddressBuffer, BufferUsageBits::Storage) ||
        !validateBuffer(desc.scratchBuffer, BufferUsageBits::Storage) ||
        (desc.buildInfoCountBuffer != nullptr &&
         !validateBuffer(desc.buildInfoCountBuffer, BufferUsageBits::Storage)) ||
        (desc.destinationSizeBuffer != nullptr &&
         !validateBuffer(desc.destinationSizeBuffer, BufferUsageBits::Storage))) {
        return makeError(Error::InvalidArgument);
    }
    if (desc.destinationMode == ClusterAccelerationStructureDestinationMode::Implicit) {
        if (!validateBuffer(
                desc.destinationStorageBuffer,
                BufferUsageBits::AccelerationStructureStorage)) {
            return makeError(Error::InvalidArgument);
        }
    } else if (!hasFlag(
                   desc.destinationAddressBuffer->desc().usage,
                   BufferUsageBits::AccelerationStructureBuildInput)) {
        return makeError(Error::InvalidArgument);
    }

    const uint64_t buildInfoSize = desc.buildInfoSize != 0
        ? desc.buildInfoSize
        : static_cast<uint64_t>(desc.maxAccelerationStructureCount) *
            desc.buildInfoStride;
    const uint64_t destinationAddressSize = desc.destinationAddressSize != 0
        ? desc.destinationAddressSize
        : static_cast<uint64_t>(desc.maxAccelerationStructureCount) *
            desc.destinationAddressStride;
    const uint64_t destinationSizeSize = desc.destinationSizeBuffer != nullptr
        ? (desc.destinationSizeSize != 0
            ? desc.destinationSizeSize
            : static_cast<uint64_t>(desc.maxAccelerationStructureCount) *
                desc.destinationSizeStride)
        : 0;
    auto rangeValid = [](const Buffer& buffer, uint64_t offset, uint64_t size) {
        return offset <= buffer.desc().size && size <= buffer.desc().size - offset;
    };
    if (!rangeValid(*desc.buildInfoBuffer, desc.buildInfoBufferOffset, buildInfoSize) ||
        !rangeValid(
            *desc.destinationAddressBuffer,
            desc.destinationAddressBufferOffset,
            destinationAddressSize) ||
        !rangeValid(*desc.scratchBuffer, desc.scratchBufferOffset, 1) ||
        (desc.buildInfoCountBuffer != nullptr &&
         !rangeValid(
             *desc.buildInfoCountBuffer,
             desc.buildInfoCountBufferOffset,
             sizeof(uint32_t))) ||
        (desc.destinationSizeBuffer != nullptr &&
         !rangeValid(
             *desc.destinationSizeBuffer,
             desc.destinationSizeBufferOffset,
             destinationSizeSize))) {
        return makeError(Error::InvalidArgument);
    }

    const uint64_t buildInfoAddress = desc.buildInfoBuffer->deviceAddress();
    const uint64_t destinationAddress = desc.destinationAddressBuffer->deviceAddress();
    const uint64_t scratchBase = desc.scratchBuffer->deviceAddress();
    const uint64_t destinationStorageAddress =
        desc.destinationStorageBuffer != nullptr
        ? desc.destinationStorageBuffer->deviceAddress()
        : 0;
    const uint64_t countAddress = desc.buildInfoCountBuffer != nullptr
        ? desc.buildInfoCountBuffer->deviceAddress()
        : 0;
    const uint64_t sizeAddress = desc.destinationSizeBuffer != nullptr
        ? desc.destinationSizeBuffer->deviceAddress()
        : 0;
    if (buildInfoAddress == 0 || destinationAddress == 0 || scratchBase == 0 ||
        (desc.destinationMode == ClusterAccelerationStructureDestinationMode::Implicit &&
         destinationStorageAddress == 0) ||
        (desc.buildInfoCountBuffer != nullptr && countAddress == 0) ||
        (desc.destinationSizeBuffer != nullptr && sizeAddress == 0)) {
        return makeError(Error::Failure);
    }

    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV,
    };
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &properties,
    };
    vkGetPhysicalDeviceProperties2(impl_->device->physicalDevice, &properties2);
    const uint64_t scratchAlignment = std::max<uint64_t>(
        1,
        properties.clusterScratchByteAlignment);
    const uint64_t unalignedScratchAddress = scratchBase + desc.scratchBufferOffset;
    const uint64_t scratchAddress =
        (unalignedScratchAddress + scratchAlignment - 1u) & ~(scratchAlignment - 1u);
    if (scratchAddress < scratchBase ||
        scratchAddress >= scratchBase + desc.scratchBuffer->desc().size) {
        return makeError(Error::InvalidArgument);
    }

    VkClusterAccelerationStructureClustersBottomLevelInputNV bottomLevelInput{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
        .maxTotalClusterCount = desc.maxTotalClusterCount,
        .maxClusterCountPerAccelerationStructure =
            desc.maxClusterCountPerAccelerationStructure,
    };
    VkClusterAccelerationStructureInputInfoNV input{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = desc.maxAccelerationStructureCount,
        .flags = toVkAccelerationStructureBuildFlags(desc.flags),
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
        .opMode = desc.destinationMode ==
                ClusterAccelerationStructureDestinationMode::Implicit
            ? VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV
            : VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
        .opInput = {.pClustersBottomLevel = &bottomLevelInput},
    };
    VkClusterAccelerationStructureCommandsInfoNV commands{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
        .input = input,
        .dstImplicitData = desc.destinationMode ==
                ClusterAccelerationStructureDestinationMode::Implicit
            ? destinationStorageAddress + desc.destinationStorageBufferOffset
            : 0,
        .scratchData = scratchAddress,
        .dstAddressesArray = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = destinationAddress + desc.destinationAddressBufferOffset,
            .stride = desc.destinationAddressStride,
            .size = destinationAddressSize,
        },
        .dstSizesArray = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = sizeAddress == 0
                ? 0
                : sizeAddress + desc.destinationSizeBufferOffset,
            .stride = desc.destinationSizeStride,
            .size = destinationSizeSize,
        },
        .srcInfosArray = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = buildInfoAddress + desc.buildInfoBufferOffset,
            .stride = desc.buildInfoStride,
            .size = buildInfoSize,
        },
        .srcInfosCount = countAddress == 0
            ? 0
            : countAddress + desc.buildInfoCountBufferOffset,
    };

    const VkMemoryBarrier2 inputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT |
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT |
            VK_ACCESS_2_MEMORY_READ_BIT |
            VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    };
    const VkDependencyInfo inputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &inputBarrier,
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &inputDependency);
    vkCmdBuildClusterAccelerationStructureIndirectNV(impl_->commandBuffer, &commands);

    const VkMemoryBarrier2 outputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            VK_ACCESS_2_SHADER_READ_BIT,
    };
    const VkDependencyInfo outputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &outputBarrier,
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &outputDependency);
    return {};
#endif
}

Result CommandBuffer::buildPartitionedAccelerationStructure(
    const PartitionedAccelerationStructureBuildDesc& desc)
{
#ifndef VK_NV_partitioned_acceleration_structure
    (void)desc;
    return makeError(Error::Unsupported);
#else
    if (impl_ == nullptr || impl_->device == nullptr ||
        !impl_->device->partitionedAccelerationStructureEnabled ||
        vkCmdBuildPartitionedAccelerationStructuresNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    if (desc.destination == nullptr || desc.destination->impl_ == nullptr ||
        !desc.destination->valid() ||
        desc.destination->impl_->device != impl_->device ||
        desc.instanceBuffer == nullptr || desc.instanceBuffer->impl_ == nullptr ||
        desc.instanceBuffer->impl_->device != impl_->device ||
        desc.scratchBuffer == nullptr || desc.scratchBuffer->impl_ == nullptr ||
        desc.scratchBuffer->impl_->device != impl_->device ||
        desc.instanceCount == 0 ||
        desc.instanceCount != desc.destination->impl_->desc.inputs.instanceCount ||
        !hasFlag(
            desc.instanceBuffer->desc().usage,
            BufferUsageBits::AccelerationStructureBuildInput) ||
        !hasFlag(desc.scratchBuffer->desc().usage, BufferUsageBits::Storage)) {
        return makeError(Error::InvalidArgument);
    }

    const uint64_t instanceAddress = desc.instanceBuffer->deviceAddress();
    const uint64_t scratchBase = desc.scratchBuffer->deviceAddress();
    const uint64_t operationAddress =
        desc.destination->impl_->operationBuffer->deviceAddress();
    const uint64_t operationCountAddress =
        desc.destination->impl_->operationCountBuffer->deviceAddress();
    if (instanceAddress == 0 || scratchBase == 0 || operationAddress == 0 ||
        operationCountAddress == 0) {
        return makeError(Error::Failure);
    }

    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    };
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &properties,
    };
    vkGetPhysicalDeviceProperties2(impl_->device->physicalDevice, &properties2);
    const uint64_t scratchAlignment = std::max<uint64_t>(
        1,
        properties.minAccelerationStructureScratchOffsetAlignment);
    const uint64_t unalignedScratchAddress = scratchBase + desc.scratchBufferOffset;
    const uint64_t scratchAddress =
        (unalignedScratchAddress + scratchAlignment - 1u) & ~(scratchAlignment - 1u);
    const uint64_t alignedScratchOffset = scratchAddress - scratchBase;
    if (alignedScratchOffset >= desc.scratchBuffer->desc().size ||
        desc.destination->impl_->desc.sizes.buildScratchSize >
            desc.scratchBuffer->desc().size - alignedScratchOffset) {
        return makeError(Error::InvalidArgument);
    }

    VkBuildPartitionedAccelerationStructureIndirectCommandNV operation{
        .opType = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV,
        .argCount = desc.instanceCount,
        .argData = VkStridedDeviceAddressNV{
            .startAddress = instanceAddress,
            .strideInBytes = sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV),
        },
    };
    void* mappedOperation = desc.destination->impl_->operationBuffer->map();
    void* mappedOperationCount = desc.destination->impl_->operationCountBuffer->map();
    if (mappedOperation == nullptr || mappedOperationCount == nullptr) {
        if (mappedOperation != nullptr) {
            desc.destination->impl_->operationBuffer->unmap();
        }
        if (mappedOperationCount != nullptr) {
            desc.destination->impl_->operationCountBuffer->unmap();
        }
        return makeError(Error::Failure);
    }
    std::memcpy(mappedOperation, &operation, sizeof(operation));
    const uint32_t operationCount = 1;
    std::memcpy(mappedOperationCount, &operationCount, sizeof(operationCount));
    desc.destination->impl_->operationBuffer->flush(0, sizeof(operation));
    desc.destination->impl_->operationCountBuffer->flush(0, sizeof(operationCount));
    desc.destination->impl_->operationBuffer->unmap();
    desc.destination->impl_->operationCountBuffer->unmap();

    const PartitionedAccelerationStructureBuildInputs& inputs =
        desc.destination->impl_->desc.inputs;
    VkPartitionedAccelerationStructureFlagsNV partitionedFlags{
        .sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV,
        .enablePartitionTranslation = inputs.allowPartitionTranslation ? VK_TRUE : VK_FALSE,
    };
    VkPartitionedAccelerationStructureInstancesInputNV inputInfo{
        .sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV,
        .pNext = &partitionedFlags,
        .flags = toVkAccelerationStructureBuildFlags(inputs.flags),
        .instanceCount = inputs.instanceCount,
        .maxInstancePerPartitionCount = inputs.maxInstancePerPartitionCount,
        .partitionCount = inputs.partitionCount,
        .maxInstanceInGlobalPartitionCount = inputs.maxInstanceInGlobalPartitionCount,
    };
    VkBuildPartitionedAccelerationStructureInfoNV buildInfo{
        .sType = VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV,
        .input = inputInfo,
        .srcAccelerationStructureData = 0,
        .dstAccelerationStructureData = desc.destination->impl_->address,
        .scratchData = scratchAddress,
        .srcInfos = operationAddress,
        .srcInfosCount = operationCountAddress,
    };
    const VkMemoryBarrier2 inputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT |
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT |
            VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    };
    const VkDependencyInfo inputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &inputBarrier,
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &inputDependency);
    vkCmdBuildPartitionedAccelerationStructuresNV(impl_->commandBuffer, &buildInfo);

    const VkMemoryBarrier2 outputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    const VkDependencyInfo outputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &outputBarrier,
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &outputDependency);
    return {};
#endif
}

CommandPool::CommandPool(std::unique_ptr<detail::CommandPoolImpl> impl)
    : impl_(std::move(impl))
{
}

CommandPool::~CommandPool()
{
    if (impl_ != nullptr && impl_->pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(impl_->device->device, impl_->pool, nullptr);
        impl_->pool = VK_NULL_HANDLE;
    }
}

CommandPool::CommandPool(CommandPool&&) noexcept = default;
CommandPool& CommandPool::operator=(CommandPool&&) noexcept = default;

Result CommandPool::reset()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromVk(vkResetCommandPool(impl_->device->device, impl_->pool, 0));
}

Result CommandPool::createCommandBuffer(std::unique_ptr<CommandBuffer>& outCommandBuffer)
{
    outCommandBuffer.reset();
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    VkCommandBufferAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = impl_->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    const VkResult result = vkAllocateCommandBuffers(impl_->device->device, &allocateInfo, &commandBuffer);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto commandBufferImpl = std::make_unique<detail::CommandBufferImpl>();
    commandBufferImpl->device = impl_->device;
    commandBufferImpl->pool = impl_->pool;
    commandBufferImpl->commandBuffer = commandBuffer;
    commandBufferImpl->queueFamilyIndex = impl_->queueFamilyIndex;
    outCommandBuffer.reset(new CommandBuffer(std::move(commandBufferImpl)));
    return {};
}

Swapchain::Swapchain(std::unique_ptr<detail::SwapchainImpl> impl)
    : impl_(std::move(impl))
{
}

Swapchain::~Swapchain() = default;
Swapchain::Swapchain(Swapchain&&) noexcept = default;
Swapchain& Swapchain::operator=(Swapchain&&) noexcept = default;

uint32_t Swapchain::imageCount() const
{
    return impl_ != nullptr ? static_cast<uint32_t>(impl_->textures.size()) : 0;
}

uint32_t Swapchain::width() const
{
    return impl_ != nullptr ? impl_->width : 0;
}

uint32_t Swapchain::height() const
{
    return impl_ != nullptr ? impl_->height : 0;
}

Format Swapchain::format() const
{
    return impl_ != nullptr ? impl_->format : Format::Unknown;
}

Texture* Swapchain::texture(uint32_t imageIndex)
{
    if (impl_ == nullptr || imageIndex >= impl_->textures.size()) {
        return nullptr;
    }
    return impl_->textures[imageIndex].get();
}

Result Swapchain::acquireNextImage(SwapchainSemaphore& semaphore, uint32_t& imageIndex)
{
    if (impl_ == nullptr || semaphore.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    const VkResult result = vkAcquireNextImageKHR(
        impl_->device->device,
        impl_->swapchain,
        kAcquireTimeoutNanoseconds,
        semaphore.impl_->semaphore,
        VK_NULL_HANDLE,
        &imageIndex);
    if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
        return makeError(Error::OutOfDate);
    }
    return resultFromVk(result);
}

Result Swapchain::present(Queue& queue, uint32_t imageIndex, SwapchainSemaphore& waitSemaphore)
{
    if (impl_ == nullptr || queue.impl_ == nullptr || waitSemaphore.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    const VkSemaphore wait = waitSemaphore.impl_->semaphore;
    VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait,
        .swapchainCount = 1,
        .pSwapchains = &impl_->swapchain,
        .pImageIndices = &imageIndex,
    };

    const VkResult result = vkQueuePresentKHR(queue.impl_->queue, &presentInfo);
    if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
        return makeError(Error::OutOfDate);
    }
    return resultFromVk(result);
}

Device::Device(std::unique_ptr<detail::DeviceImpl> impl)
    : impl_(std::move(impl))
{
}

Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

const DeviceCapabilities& Device::capabilities() const
{
    static const DeviceCapabilities emptyCapabilities;
    return impl_ != nullptr ? impl_->capabilities : emptyCapabilities;
}

Result Device::queryRayTracingAccelerationStructureProperties(
    RayTracingAccelerationStructureProperties& outProperties) const
{
    outProperties = {};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->rayTracingAccelerationStructureEnabled ||
        !impl_->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    };
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &properties,
    };
    vkGetPhysicalDeviceProperties2(impl_->physicalDevice, &properties2);
    outProperties = RayTracingAccelerationStructureProperties{
        .scratchAlignment = std::max<uint64_t>(
            1,
            properties.minAccelerationStructureScratchOffsetAlignment),
        .instanceBufferAlignment = 16,
        .instanceRecordSize = sizeof(RayTracingGpuInstance),
    };
    return {};
}

Result Device::queryRayTracingAccelerationStructureBuildSizes(
    const RayTracingAccelerationStructureBuildInputs& inputs,
    RayTracingAccelerationStructureBuildSizes& outSizes) const
{
    outSizes = {};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->rayTracingAccelerationStructureEnabled ||
        !impl_->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    activateVolkDevice(impl_->device);
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<uint32_t> primitiveCounts;
    if (inputs.type == RayTracingAccelerationStructureType::BottomLevel) {
        if (inputs.geometries == nullptr || inputs.geometryCount == 0 ||
            inputs.instanceCount != 0) {
            return makeError(Error::InvalidArgument);
        }
        geometries.reserve(inputs.geometryCount);
        primitiveCounts.reserve(inputs.geometryCount);
        for (uint32_t index = 0; index < inputs.geometryCount; ++index) {
            const RayTracingTriangleGeometryDesc& source = inputs.geometries[index];
            if (source.vertexBuffer == nullptr || source.vertexBuffer->impl_ == nullptr ||
                source.vertexBuffer->impl_->device != impl_.get() ||
                source.vertexCount == 0 || source.vertexStride == 0 ||
                source.primitiveCount == 0 || source.vertexFormat == Format::Unknown ||
                source.vertexOffset >= source.vertexBuffer->desc().size ||
                !hasFlag(
                    source.vertexBuffer->desc().usage,
                    BufferUsageBits::AccelerationStructureBuildInput)) {
                return makeError(Error::InvalidArgument);
            }
            const VkDeviceAddress vertexAddress = source.vertexBuffer->deviceAddress();
            const VkFormat vertexFormat = toVkFormat(source.vertexFormat);
            if (vertexAddress == 0 || vertexFormat == VK_FORMAT_UNDEFINED) {
                return makeError(Error::InvalidArgument);
            }

            VkDeviceAddress indexAddress = 0;
            if (source.indexType != RayTracingIndexType::None) {
                if (source.indexBuffer == nullptr || source.indexBuffer->impl_ == nullptr ||
                    source.indexBuffer->impl_->device != impl_.get() ||
                    source.indexOffset >= source.indexBuffer->desc().size ||
                    !hasFlag(
                        source.indexBuffer->desc().usage,
                        BufferUsageBits::AccelerationStructureBuildInput)) {
                    return makeError(Error::InvalidArgument);
                }
                indexAddress = source.indexBuffer->deviceAddress();
                if (indexAddress == 0) {
                    return makeError(Error::Failure);
                }
            }

            VkAccelerationStructureGeometryTrianglesDataKHR triangles{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .vertexFormat = vertexFormat,
                .vertexData = {.deviceAddress = vertexAddress + source.vertexOffset},
                .vertexStride = source.vertexStride,
                .maxVertex = source.vertexCount - 1,
                .indexType = toVkRayTracingIndexType(source.indexType),
                .indexData = {.deviceAddress = indexAddress == 0 ? 0 : indexAddress + source.indexOffset},
            };
            VkAccelerationStructureGeometryKHR geometry{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                .geometry = {.triangles = triangles},
                .flags = toVkGeometryFlags(source.flags),
            };
            geometries.push_back(geometry);
            primitiveCounts.push_back(source.primitiveCount);
        }
    } else {
        if (inputs.geometries != nullptr || inputs.geometryCount != 0 ||
            inputs.instanceCount == 0) {
            return makeError(Error::InvalidArgument);
        }
        VkAccelerationStructureGeometryInstancesDataKHR instances{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        };
        geometries.push_back(VkAccelerationStructureGeometryKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
            .geometry = {.instances = instances},
        });
        primitiveCounts.push_back(inputs.instanceCount);
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = toVkAccelerationStructureType(inputs.type),
        .flags = toVkAccelerationStructureBuildFlags(inputs.flags),
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = static_cast<uint32_t>(geometries.size()),
        .pGeometries = geometries.data(),
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetAccelerationStructureBuildSizesKHR(
        impl_->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        primitiveCounts.data(),
        &sizes);
    if (sizes.accelerationStructureSize == 0 || sizes.buildScratchSize == 0) {
        return makeError(Error::Failure);
    }
    outSizes = RayTracingAccelerationStructureBuildSizes{
        .accelerationStructureSize = sizes.accelerationStructureSize,
        .buildScratchSize = sizes.buildScratchSize,
        .updateScratchSize = sizes.updateScratchSize,
    };
    return {};
}

Result Device::createRayTracingAccelerationStructure(
    const RayTracingAccelerationStructureDesc& desc,
    std::unique_ptr<RayTracingAccelerationStructure>& outAccelerationStructure)
{
    outAccelerationStructure.reset();
    if (impl_ == nullptr || desc.size == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->rayTracingAccelerationStructureEnabled ||
        !impl_->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    std::unique_ptr<Buffer> storage;
    Result result = createBuffer(
        BufferDesc{
            .size = desc.size,
            .usage = BufferUsageBits::AccelerationStructureStorage |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        },
        storage);
    if (!result) {
        return result;
    }

    activateVolkDevice(impl_->device);
    VkAccelerationStructureCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = storage->impl_->buffer,
        .size = desc.size,
        .type = toVkAccelerationStructureType(desc.type),
    };
    VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
    const VkResult vkResult = vkCreateAccelerationStructureKHR(
        impl_->device,
        &createInfo,
        nullptr,
        &accelerationStructure);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = accelerationStructure,
    };
    const VkDeviceAddress address = vkGetAccelerationStructureDeviceAddressKHR(
        impl_->device,
        &addressInfo);
    if (address == 0) {
        vkDestroyAccelerationStructureKHR(impl_->device, accelerationStructure, nullptr);
        return makeError(Error::Failure);
    }

    auto accelerationStructureImpl =
        std::make_unique<detail::RayTracingAccelerationStructureImpl>();
    accelerationStructureImpl->device = impl_.get();
    accelerationStructureImpl->desc = desc;
    accelerationStructureImpl->storage = std::move(storage);
    accelerationStructureImpl->accelerationStructure = accelerationStructure;
    accelerationStructureImpl->address = address;
    outAccelerationStructure.reset(
        new RayTracingAccelerationStructure(std::move(accelerationStructureImpl)));
    return {};
}

Result Device::createRayTracingInstanceBuffer(
    const RayTracingInstanceDesc* instances,
    uint32_t instanceCount,
    std::unique_ptr<Buffer>& outBuffer)
{
    outBuffer.reset();
    if (impl_ == nullptr || instances == nullptr || instanceCount == 0) {
        return makeError(Error::InvalidArgument);
    }
    Result result = createBuffer(
        BufferDesc{
            .size = static_cast<uint64_t>(instanceCount) * sizeof(RayTracingGpuInstance),
            .structureStride = sizeof(RayTracingGpuInstance),
            .usage = BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        },
        outBuffer);
    if (!result) {
        return result;
    }
    result = writeRayTracingInstances(*outBuffer, instances, instanceCount);
    if (!result) {
        outBuffer.reset();
    }
    return result;
}

Result Device::writeRayTracingInstances(
    Buffer& buffer,
    const RayTracingInstanceDesc* instances,
    uint32_t instanceCount)
{
    if (impl_ == nullptr || buffer.impl_ == nullptr || buffer.impl_->device != impl_.get() ||
        instances == nullptr || instanceCount == 0 ||
        !hasFlag(buffer.desc().usage, BufferUsageBits::AccelerationStructureBuildInput) ||
        static_cast<uint64_t>(instanceCount) * sizeof(RayTracingGpuInstance) >
            buffer.desc().size) {
        return makeError(Error::InvalidArgument);
    }

    static_assert(sizeof(RayTracingGpuInstance) == sizeof(VkAccelerationStructureInstanceKHR));
    static_assert(
        offsetof(RayTracingGpuInstance, accelerationStructureReference) ==
        offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference));
    std::vector<RayTracingGpuInstance> encoded(instanceCount);
    for (uint32_t index = 0; index < instanceCount; ++index) {
        const RayTracingInstanceDesc& source = instances[index];
        if (source.bottomLevel == nullptr || source.bottomLevel->impl_ == nullptr ||
            source.bottomLevel->impl_->device != impl_.get() ||
            source.bottomLevel->impl_->desc.type !=
                RayTracingAccelerationStructureType::BottomLevel ||
            !source.bottomLevel->valid() || source.customIndex > 0x00ffffffu ||
            source.shaderBindingTableRecordOffset > 0x00ffffffu) {
            return makeError(Error::InvalidArgument);
        }
        RayTracingGpuInstance& destination = encoded[index];
        std::memcpy(
            destination.transform,
            source.transform,
            sizeof(destination.transform));
        destination.customIndexAndMask =
            (source.customIndex & 0x00ffffffu) |
            (static_cast<uint32_t>(source.mask) << 24u);
        destination.shaderBindingTableRecordOffsetAndFlags =
            (source.shaderBindingTableRecordOffset & 0x00ffffffu) |
            (static_cast<uint32_t>(toVkInstanceFlags(source.flags)) << 24u);
        destination.accelerationStructureReference = source.bottomLevel->impl_->address;
    }

    void* mapped = buffer.map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }
    const uint64_t byteSize = static_cast<uint64_t>(encoded.size()) * sizeof(encoded[0]);
    std::memcpy(mapped, encoded.data(), static_cast<size_t>(byteSize));
    buffer.flush(0, byteSize);
    buffer.unmap();
    return {};
}

Result Device::queryClusterAccelerationStructureProperties(
    ClusterAccelerationStructureProperties& outProperties) const
{
    outProperties = {};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->clusterAccelerationStructureEnabled ||
        !impl_->capabilities.clusterAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
#ifndef VK_NV_cluster_acceleration_structure
    return makeError(Error::Unsupported);
#else
    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV,
    };
    VkPhysicalDeviceProperties2 deviceProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &properties,
    };
    vkGetPhysicalDeviceProperties2(impl_->physicalDevice, &deviceProperties);
    if (properties.clusterByteAlignment == 0 ||
        properties.clusterScratchByteAlignment == 0) {
        return makeError(Error::Failure);
    }
    outProperties = ClusterAccelerationStructureProperties{
        .clusterStorageAlignment = properties.clusterByteAlignment,
        .bottomLevelStorageAlignment = properties.clusterBottomLevelByteAlignment,
        .scratchAlignment = properties.clusterScratchByteAlignment,
        .triangleBuildInfoSize =
            sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
        .bottomLevelBuildInfoSize =
            sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
    };
    return {};
#endif
}

Result Device::queryClusterAccelerationStructureTriangleBuildSizes(
    const ClusterAccelerationStructureTriangleBuildSizesDesc& desc,
    ClusterAccelerationStructureBuildSizes& outSizes) const
{
    outSizes = {};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->clusterAccelerationStructureEnabled ||
        !impl_->capabilities.clusterAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
    if (desc.maxClusterTriangleCount == 0 ||
        desc.maxClusterVertexCount == 0 ||
        desc.maxClusterUniqueGeometryCount == 0 ||
        desc.maxTotalTriangleCount == 0 ||
        desc.maxTotalVertexCount == 0 ||
        desc.maxAccelerationStructureCount == 0 ||
        desc.vertexFormat == Format::Unknown) {
        return makeError(Error::InvalidArgument);
    }
#ifndef VK_NV_cluster_acceleration_structure
    return makeError(Error::Unsupported);
#else
    activateVolkDevice(impl_->device);
    if (vkGetClusterAccelerationStructureBuildSizesNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    const VkFormat vertexFormat = toVkFormat(desc.vertexFormat);
    if (vertexFormat == VK_FORMAT_UNDEFINED) {
        return makeError(Error::InvalidArgument);
    }
    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV,
        .vertexFormat = vertexFormat,
        .maxGeometryIndexValue = desc.maxGeometryIndexValue,
        .maxClusterUniqueGeometryCount = desc.maxClusterUniqueGeometryCount,
        .maxClusterTriangleCount = desc.maxClusterTriangleCount,
        .maxClusterVertexCount = desc.maxClusterVertexCount,
        .maxTotalTriangleCount = desc.maxTotalTriangleCount,
        .maxTotalVertexCount = desc.maxTotalVertexCount,
        .minPositionTruncateBitCount = desc.minPositionTruncateBitCount,
    };
    VkClusterAccelerationStructureInputInfoNV inputInfo{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = desc.maxAccelerationStructureCount,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
        .opInput = {.pTriangleClusters = &triangleInput},
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetClusterAccelerationStructureBuildSizesNV(impl_->device, &inputInfo, &sizes);
    outSizes = ClusterAccelerationStructureBuildSizes{
        .accelerationStructureSize = sizes.accelerationStructureSize,
        .updateScratchSize = sizes.updateScratchSize,
        .buildScratchSize = sizes.buildScratchSize,
    };
    return {};
#endif
}

Result Device::queryClusterAccelerationStructureBottomLevelBuildSizes(
    const ClusterAccelerationStructureBottomLevelBuildSizesDesc& desc,
    ClusterAccelerationStructureBuildSizes& outSizes) const
{
    outSizes = {};
    if (impl_ == nullptr ||
        desc.maxClusterCountPerAccelerationStructure == 0 ||
        desc.maxTotalClusterCount == 0 ||
        desc.maxAccelerationStructureCount == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->clusterAccelerationStructureEnabled ||
        !impl_->capabilities.clusterAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
#ifndef VK_NV_cluster_acceleration_structure
    return makeError(Error::Unsupported);
#else
    activateVolkDevice(impl_->device);
    if (vkGetClusterAccelerationStructureBuildSizesNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    VkClusterAccelerationStructureClustersBottomLevelInputNV bottomLevelInput{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
        .maxTotalClusterCount = desc.maxTotalClusterCount,
        .maxClusterCountPerAccelerationStructure =
            desc.maxClusterCountPerAccelerationStructure,
    };
    VkClusterAccelerationStructureInputInfoNV inputInfo{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = desc.maxAccelerationStructureCount,
        .flags = toVkAccelerationStructureBuildFlags(desc.flags),
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
        .opInput = {.pClustersBottomLevel = &bottomLevelInput},
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetClusterAccelerationStructureBuildSizesNV(impl_->device, &inputInfo, &sizes);
    if (sizes.accelerationStructureSize == 0 || sizes.buildScratchSize == 0) {
        return makeError(Error::Failure);
    }
    outSizes = ClusterAccelerationStructureBuildSizes{
        .accelerationStructureSize = sizes.accelerationStructureSize,
        .updateScratchSize = sizes.updateScratchSize,
        .buildScratchSize = sizes.buildScratchSize,
    };
    return {};
#endif
}

Result Device::queryPartitionedAccelerationStructureBuildSizes(
    const PartitionedAccelerationStructureBuildInputs& inputs,
    PartitionedAccelerationStructureBuildSizes& outSizes) const
{
    outSizes = {};
    if (impl_ == nullptr || inputs.instanceCount == 0 ||
        inputs.partitionCount == 0 || inputs.maxInstancePerPartitionCount == 0 ||
        inputs.maxOperationCount == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->partitionedAccelerationStructureEnabled ||
        !impl_->capabilities.partitionedAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
#ifndef VK_NV_partitioned_acceleration_structure
    return makeError(Error::Unsupported);
#else
    activateVolkDevice(impl_->device);
    if (vkGetPartitionedAccelerationStructuresBuildSizesNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    VkPartitionedAccelerationStructureFlagsNV partitionedFlags{
        .sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV,
        .enablePartitionTranslation = inputs.allowPartitionTranslation ? VK_TRUE : VK_FALSE,
    };
    VkPartitionedAccelerationStructureInstancesInputNV inputInfo{
        .sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV,
        .pNext = &partitionedFlags,
        .flags = toVkAccelerationStructureBuildFlags(inputs.flags),
        .instanceCount = inputs.instanceCount,
        .maxInstancePerPartitionCount = inputs.maxInstancePerPartitionCount,
        .partitionCount = inputs.partitionCount,
        .maxInstanceInGlobalPartitionCount = inputs.maxInstanceInGlobalPartitionCount,
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetPartitionedAccelerationStructuresBuildSizesNV(impl_->device, &inputInfo, &sizes);
    if (sizes.accelerationStructureSize == 0 || sizes.buildScratchSize == 0) {
        return makeError(Error::Failure);
    }
    outSizes = PartitionedAccelerationStructureBuildSizes{
        .accelerationStructureSize = sizes.accelerationStructureSize,
        .updateScratchSize = sizes.updateScratchSize,
        .buildScratchSize = sizes.buildScratchSize,
        .operationInfoSize = static_cast<uint64_t>(inputs.maxOperationCount) *
            sizeof(VkBuildPartitionedAccelerationStructureIndirectCommandNV),
        .operationCountSize = sizeof(uint32_t),
        .instanceWriteInfoSize = static_cast<uint64_t>(inputs.instanceCount) *
            sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV),
        .instanceUpdateInfoSize = inputs.allowInstanceUpdate
            ? static_cast<uint64_t>(inputs.instanceCount) *
                sizeof(VkPartitionedAccelerationStructureUpdateInstanceDataNV)
            : 0,
        .partitionWriteInfoSize = inputs.allowPartitionTranslation
            ? static_cast<uint64_t>(inputs.partitionCount + 1u) *
                sizeof(VkPartitionedAccelerationStructureWritePartitionTranslationDataNV)
            : 0,
    };
    return {};
#endif
}

Result Device::createPartitionedAccelerationStructure(
    const PartitionedAccelerationStructureDesc& desc,
    std::unique_ptr<PartitionedAccelerationStructure>& outAccelerationStructure)
{
    outAccelerationStructure.reset();
    if (impl_ == nullptr || desc.sizes.accelerationStructureSize == 0 ||
        desc.sizes.operationInfoSize == 0 || desc.sizes.operationCountSize == 0) {
        return makeError(Error::InvalidArgument);
    }
    PartitionedAccelerationStructureBuildSizes expectedSizes;
    Result result = queryPartitionedAccelerationStructureBuildSizes(
        desc.inputs,
        expectedSizes);
    if (!result) {
        return result;
    }
    if (desc.sizes.accelerationStructureSize < expectedSizes.accelerationStructureSize ||
        desc.sizes.operationInfoSize < expectedSizes.operationInfoSize ||
        desc.sizes.operationCountSize < expectedSizes.operationCountSize) {
        return makeError(Error::InvalidArgument);
    }

    auto implementation = std::make_unique<detail::PartitionedAccelerationStructureImpl>();
    implementation->device = impl_.get();
    implementation->desc = desc;
    result = createBuffer(
        BufferDesc{
            .size = desc.sizes.accelerationStructureSize,
            .usage = BufferUsageBits::AccelerationStructureStorage |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::Device,
        },
        implementation->storage);
    if (!result) {
        return result;
    }
    result = createBuffer(
        BufferDesc{
            .size = desc.sizes.operationInfoSize,
            .usage = BufferUsageBits::Storage |
                BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload,
        },
        implementation->operationBuffer);
    if (!result) {
        return result;
    }
    result = createBuffer(
        BufferDesc{
            .size = desc.sizes.operationCountSize,
            .usage = BufferUsageBits::Storage |
                BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload,
        },
        implementation->operationCountBuffer);
    if (!result) {
        return result;
    }
    implementation->address = implementation->storage->deviceAddress();
    if (implementation->address == 0) {
        return makeError(Error::Failure);
    }
    outAccelerationStructure.reset(
        new PartitionedAccelerationStructure(std::move(implementation)));
    return {};
}

Result Device::createPartitionedAccelerationStructureInstanceBuffer(
    const PartitionedAccelerationStructureInstanceDesc* instances,
    uint32_t instanceCount,
    std::unique_ptr<Buffer>& outBuffer)
{
    outBuffer.reset();
    if (impl_ == nullptr || instances == nullptr || instanceCount == 0) {
        return makeError(Error::InvalidArgument);
    }
#ifndef VK_NV_partitioned_acceleration_structure
    return makeError(Error::Unsupported);
#else
    std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> encoded(instanceCount);
    for (uint32_t index = 0; index < instanceCount; ++index) {
        const PartitionedAccelerationStructureInstanceDesc& source = instances[index];
        if (source.bottomLevel == nullptr || source.bottomLevel->impl_ == nullptr ||
            source.bottomLevel->impl_->device != impl_.get() ||
            source.bottomLevel->desc().type !=
                RayTracingAccelerationStructureType::BottomLevel ||
            !source.bottomLevel->valid() || source.customIndex > 0x00ffffffu ||
            source.shaderBindingTableRecordOffset > 0x00ffffffu) {
            return makeError(Error::InvalidArgument);
        }
        VkPartitionedAccelerationStructureWriteInstanceDataNV& destination = encoded[index];
        std::memcpy(destination.transform.matrix, source.transform, sizeof(source.transform));
        destination.instanceID = source.customIndex;
        destination.instanceMask = source.mask;
        destination.instanceContributionToHitGroupIndex =
            source.shaderBindingTableRecordOffset;
        destination.instanceFlags = static_cast<VkPartitionedAccelerationStructureInstanceFlagsNV>(
            toVkInstanceFlags(source.flags));
        destination.instanceIndex = source.instanceIndex;
        destination.partitionIndex = source.partitionIndex;
        destination.accelerationStructure = source.bottomLevel->impl_->address;
    }
    Result result = createBuffer(
        BufferDesc{
            .size = static_cast<uint64_t>(encoded.size()) * sizeof(encoded[0]),
            .structureStride = sizeof(encoded[0]),
            .usage = BufferUsageBits::Storage |
                BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload,
        },
        outBuffer);
    if (!result) {
        return result;
    }
    void* mapped = outBuffer->map();
    if (mapped == nullptr) {
        outBuffer.reset();
        return makeError(Error::Failure);
    }
    const uint64_t byteSize = static_cast<uint64_t>(encoded.size()) * sizeof(encoded[0]);
    std::memcpy(mapped, encoded.data(), static_cast<size_t>(byteSize));
    outBuffer->flush(0, byteSize);
    outBuffer->unmap();
    return {};
#endif
}

Queue* Device::getQueue(QueueType type, uint32_t index)
{
    if (impl_ == nullptr) {
        return nullptr;
    }

    uint32_t seen = 0;
    for (const std::unique_ptr<Queue>& queue : impl_->queues) {
        if (queue->type() == type) {
            if (seen == index) {
                return queue.get();
            }
            ++seen;
        }
    }
    return nullptr;
}

Result Device::waitIdle()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);
    return resultFromVk(vkDeviceWaitIdle(impl_->device));
}

Result Device::createSwapchain(const SwapchainDesc& desc, std::unique_ptr<Swapchain>& outSwapchain)
{
    outSwapchain.reset();
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    auto swapchainImpl = std::make_unique<detail::SwapchainImpl>();
    swapchainImpl->device = impl_.get();
    const Result result = swapchainImpl->initialize(desc);
    if (!result) {
        return result;
    }

    outSwapchain.reset(new Swapchain(std::move(swapchainImpl)));
    return {};
}

Result Device::createCommandPool(Queue& queue, std::unique_ptr<CommandPool>& outCommandPool)
{
    outCommandPool.reset();
    if (impl_ == nullptr || queue.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    VkCommandPoolCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue.impl_->familyIndex,
    };

    VkCommandPool pool = VK_NULL_HANDLE;
    const VkResult result = vkCreateCommandPool(impl_->device, &createInfo, nullptr, &pool);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto poolImpl = std::make_unique<detail::CommandPoolImpl>();
    poolImpl->device = impl_.get();
    poolImpl->pool = pool;
    poolImpl->queueFamilyIndex = queue.impl_->familyIndex;
    outCommandPool.reset(new CommandPool(std::move(poolImpl)));
    return {};
}

Result Device::createFence(bool signaled, std::unique_ptr<Fence>& outFence)
{
    outFence.reset();
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    VkFenceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0u,
    };

    VkFence fence = VK_NULL_HANDLE;
    const VkResult result = vkCreateFence(impl_->device, &createInfo, nullptr, &fence);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto fenceImpl = std::make_unique<detail::FenceImpl>();
    fenceImpl->device = impl_.get();
    fenceImpl->fence = fence;
    outFence.reset(new Fence(std::move(fenceImpl)));
    return {};
}

Result Device::createTimestampQueryPool(
    Queue& queue,
    const TimestampQueryPoolDesc& desc,
    std::unique_ptr<TimestampQueryPool>& outQueryPool)
{
    outQueryPool.reset();
    if (impl_ == nullptr ||
        queue.impl_ == nullptr ||
        queue.impl_->device != impl_.get() ||
        desc.queryCount == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (queue.impl_->timestampValidBits == 0 ||
        impl_->capabilities.timestampPeriodNanoseconds <= 0.0) {
        return makeError(Error::Unsupported);
    }

    activateVolkDevice(impl_->device);
    VkQueryPoolCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = desc.queryCount,
    };
    VkQueryPool queryPool = VK_NULL_HANDLE;
    const VkResult result = vkCreateQueryPool(impl_->device, &createInfo, nullptr, &queryPool);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto queryPoolImpl = std::make_unique<detail::TimestampQueryPoolImpl>();
    queryPoolImpl->device = impl_.get();
    queryPoolImpl->desc = desc;
    queryPoolImpl->queryPool = queryPool;
    queryPoolImpl->queueFamilyIndex = queue.impl_->familyIndex;
    queryPoolImpl->timestampValidBits = queue.impl_->timestampValidBits;
    queryPoolImpl->timestampPeriodNanoseconds = impl_->capabilities.timestampPeriodNanoseconds;
    outQueryPool.reset(new TimestampQueryPool(std::move(queryPoolImpl)));
    return {};
}

Result Device::createSemaphore(std::unique_ptr<Semaphore>& outSemaphore)
{
    return createSemaphore(SemaphoreDesc{}, outSemaphore);
}

Result Device::createSemaphore(const SemaphoreDesc& desc, std::unique_ptr<Semaphore>& outSemaphore)
{
    outSemaphore.reset();
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    VkSemaphoreTypeCreateInfo typeCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = desc.initialValue,
    };
    VkSemaphoreCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &typeCreateInfo,
    };

    VkSemaphore semaphore = VK_NULL_HANDLE;
    const VkResult result = vkCreateSemaphore(impl_->device, &createInfo, nullptr, &semaphore);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto semaphoreImpl = std::make_unique<detail::SemaphoreImpl>();
    semaphoreImpl->device = impl_.get();
    semaphoreImpl->semaphore = semaphore;
    outSemaphore.reset(new Semaphore(std::move(semaphoreImpl)));
    return {};
}

Result Device::createSwapchainSemaphore(std::unique_ptr<SwapchainSemaphore>& outSemaphore)
{
    outSemaphore.reset();
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    VkSemaphoreCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    VkSemaphore semaphore = VK_NULL_HANDLE;
    const VkResult result = vkCreateSemaphore(impl_->device, &createInfo, nullptr, &semaphore);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto semaphoreImpl = std::make_unique<detail::SwapchainSemaphoreImpl>();
    semaphoreImpl->device = impl_.get();
    semaphoreImpl->semaphore = semaphore;
    outSemaphore.reset(new SwapchainSemaphore(std::move(semaphoreImpl)));
    return {};
}

Result Device::createBuffer(const BufferDesc& desc, std::unique_ptr<Buffer>& outBuffer)
{
    outBuffer.reset();
    if (impl_ == nullptr || desc.size == 0) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    const bool usesAccelerationStructure =
        hasFlag(desc.usage, BufferUsageBits::AccelerationStructureBuildInput) ||
        hasFlag(desc.usage, BufferUsageBits::AccelerationStructureStorage);
    if (usesAccelerationStructure && !impl_->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    const bool requestsDeviceAddress =
        hasFlag(desc.usage, BufferUsageBits::ShaderDeviceAddress) ||
        usesAccelerationStructure;
    if (requestsDeviceAddress && !impl_->bufferDeviceAddressEnabled) {
        return makeError(Error::Unsupported);
    }

    VkBufferUsageFlags usage = toVkBufferUsage(desc.usage);
    if (impl_->bufferDeviceAddressEnabled &&
        (hasFlag(desc.usage, BufferUsageBits::Constant) || hasFlag(desc.usage, BufferUsageBits::Storage))) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    const std::vector<uint32_t> queueFamilies = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = desc.size,
        .usage = usage,
        .sharingMode = queueFamilies.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = queueFamilies.size() > 1
            ? static_cast<uint32_t>(queueFamilies.size())
            : 0,
        .pQueueFamilyIndices = queueFamilies.size() > 1 ? queueFamilies.data() : nullptr,
    };

    VmaAllocationCreateInfo allocationInfo = allocationInfoForMemory(desc.memoryLocation);
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkResult result = vmaCreateBuffer(
        impl_->allocator,
        &bufferInfo,
        &allocationInfo,
        &buffer,
        &allocation,
        nullptr);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto bufferImpl = std::make_unique<detail::BufferImpl>();
    bufferImpl->device = impl_.get();
    bufferImpl->desc = desc;
    bufferImpl->buffer = buffer;
    bufferImpl->allocation = allocation;
    outBuffer.reset(new Buffer(std::move(bufferImpl)));
    return {};
}

Result Device::createBufferView(
    Buffer& buffer,
    const BufferViewDesc& desc,
    std::unique_ptr<BufferView>& outBufferView)
{
    outBufferView.reset();
    if (impl_ == nullptr || buffer.impl_ == nullptr || desc.offset >= buffer.impl_->desc.size) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);
    if (!impl_->capabilities.bindlessDescriptorHeap) {
        return makeError(Error::Unsupported);
    }

    const uint64_t availableSize = buffer.impl_->desc.size - desc.offset;
    const uint64_t viewSize = desc.size == UINT64_MAX ? availableSize : desc.size;
    if (viewSize == 0 || viewSize > availableSize) {
        return makeError(Error::InvalidArgument);
    }

    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    switch (desc.type) {
    case BufferViewType::Constant:
        if (!hasFlag(buffer.impl_->desc.usage, BufferUsageBits::Constant)) {
            return makeError(Error::InvalidArgument);
        }
        descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        break;
    case BufferViewType::Structured:
    case BufferViewType::Raw:
    case BufferViewType::ReadWriteStructured:
    case BufferViewType::ReadWriteRaw:
        if (!hasFlag(buffer.impl_->desc.usage, BufferUsageBits::Storage)) {
            return makeError(Error::InvalidArgument);
        }
        descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        break;
    }

    const uint32_t structureStride = desc.structureStride != 0
        ? desc.structureStride
        : buffer.impl_->desc.structureStride;
    if ((desc.type == BufferViewType::Structured || desc.type == BufferViewType::ReadWriteStructured) &&
        structureStride == 0) {
        return makeError(Error::InvalidArgument);
    }

    VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer.impl_->buffer,
    };
    const VkDeviceAddress bufferAddress = vkGetBufferDeviceAddress(impl_->device, &addressInfo);
    if (bufferAddress == 0) {
        return makeError(Error::Failure);
    }

    auto viewImpl = std::make_unique<detail::BufferViewImpl>();
    viewImpl->device = impl_.get();
    viewImpl->buffer = &buffer;
    viewImpl->desc = desc;
    viewImpl->desc.size = viewSize;
    viewImpl->desc.structureStride = structureStride;
    viewImpl->descriptorType = descriptorType;
    viewImpl->address = bufferAddress + desc.offset;
    viewImpl->size = viewSize;
    outBufferView.reset(new BufferView(std::move(viewImpl)));
    return {};
}

Result Device::createTexture(const TextureDesc& desc, std::unique_ptr<Texture>& outTexture)
{
    outTexture.reset();
    if (impl_ == nullptr || desc.format == Format::Unknown) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    const std::vector<uint32_t> queueFamilies = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
    VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = toVkImageType(desc.type),
        .format = toVkFormat(desc.format),
        .extent = {desc.width, desc.height, desc.depth},
        .mipLevels = desc.mipCount,
        .arrayLayers = desc.layerCount,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = toVkImageUsage(desc.usage),
        .sharingMode = queueFamilies.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = queueFamilies.size() > 1
            ? static_cast<uint32_t>(queueFamilies.size())
            : 0,
        .pQueueFamilyIndices = queueFamilies.size() > 1 ? queueFamilies.data() : nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo allocationInfo = allocationInfoForMemory(desc.memoryLocation);
    VmaAllocationInfo allocatedInfo{};
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkResult result = vmaCreateImage(
        impl_->allocator,
        &imageInfo,
        &allocationInfo,
        &image,
        &allocation,
        &allocatedInfo);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto textureImpl = std::make_unique<detail::TextureImpl>();
    textureImpl->device = impl_.get();
    textureImpl->desc = desc;
    textureImpl->image = image;
    textureImpl->memory = allocatedInfo.deviceMemory;
    textureImpl->allocation = allocation;
    textureImpl->flags = imageInfo.flags;
    textureImpl->usage = imageInfo.usage;
    textureImpl->ownsImage = true;
    outTexture.reset(new Texture(std::move(textureImpl)));
    return {};
}

Result Device::createTextureView(
    Texture& texture,
    const TextureViewDesc& desc,
    std::unique_ptr<TextureView>& outTextureView)
{
    outTextureView.reset();
    if (impl_ == nullptr || texture.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    const TextureDesc& textureDesc = texture.impl_->desc;
    const Format format = desc.format != Format::Unknown ? desc.format : textureDesc.format;
    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = texture.impl_->image,
        .viewType = toVkImageViewType(textureDesc.type),
        .format = toVkFormat(format),
        .subresourceRange = {
            .aspectMask = aspectForFormat(format),
            .baseMipLevel = desc.baseMip,
            .levelCount = desc.mipCount,
            .baseArrayLayer = desc.baseLayer,
            .layerCount = desc.layerCount,
        },
    };

    VkImageView view = VK_NULL_HANDLE;
    const VkResult result = vkCreateImageView(impl_->device, &viewInfo, nullptr, &view);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto viewImpl = std::make_unique<detail::TextureViewImpl>();
    viewImpl->device = impl_.get();
    viewImpl->texture = &texture;
    viewImpl->desc = desc;
    viewImpl->view = view;
    viewImpl->format = toVkFormat(format);
    outTextureView.reset(new TextureView(std::move(viewImpl)));
    return {};
}

Result Device::createShaderModule(const ShaderModuleDesc& desc, std::unique_ptr<ShaderModule>& outShaderModule)
{
    outShaderModule.reset();
    if (impl_ == nullptr || desc.code == nullptr || desc.byteSize == 0 || (desc.byteSize % sizeof(uint32_t)) != 0) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    VkShaderModuleCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = desc.byteSize,
        .pCode = desc.code,
    };

    VkShaderModule module = VK_NULL_HANDLE;
    const VkResult result = vkCreateShaderModule(impl_->device, &createInfo, nullptr, &module);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    profiling::registerNsightAftermathShaderBinary(desc.code, desc.byteSize);
    if (desc.debugName != nullptr && desc.debugName[0] != '\0' &&
        impl_->setDebugUtilsObjectName != nullptr) {
        const VkDebugUtilsObjectNameInfoEXT nameInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VK_OBJECT_TYPE_SHADER_MODULE,
            .objectHandle = reinterpret_cast<uint64_t>(module),
            .pObjectName = desc.debugName,
        };
        impl_->setDebugUtilsObjectName(impl_->device, &nameInfo);
    }

    auto shaderImpl = std::make_unique<detail::ShaderModuleImpl>();
    shaderImpl->device = impl_.get();
    shaderImpl->module = module;
    shaderImpl->contentHash = detail::shaderContentHash(desc);
    outShaderModule.reset(new ShaderModule(std::move(shaderImpl)));
    return {};
}

Result Device::createPipelineCache(
    const PipelineCacheDesc& desc,
    std::unique_ptr<PipelineCache>& outPipelineCache)
{
    outPipelineCache.reset();
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);

    auto cacheImpl = std::make_unique<detail::PipelineCacheImpl>();
    Result result = cacheImpl->initialize(*impl_, desc);
    if (!result) {
        return result;
    }
    outPipelineCache.reset(new PipelineCache(std::move(cacheImpl)));
    return {};
}

Result Device::createGraphicsPipeline(
    const GraphicsPipelineDesc& desc,
    std::unique_ptr<GraphicsPipeline>& outGraphicsPipeline)
{
    outGraphicsPipeline.reset();
    const bool usesTaskShader = desc.taskShader != nullptr;
    const bool usesMeshShader = desc.meshShader != nullptr;
    const bool usesVertexShader = desc.vertexShader != nullptr;
    if (impl_ == nullptr ||
        usesMeshShader == usesVertexShader ||
        (usesTaskShader && !usesMeshShader) ||
        desc.fragmentShader == nullptr ||
        desc.fragmentShader->impl_ == nullptr ||
        desc.colorFormat == Format::Unknown) {
        return makeError(Error::InvalidArgument);
    }
    if (usesVertexShader && desc.vertexShader->impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (usesTaskShader && desc.taskShader->impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (usesMeshShader && desc.meshShader->impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);
    if (usesMeshShader && !impl_->capabilities.meshShader) {
        return makeError(Error::Unsupported);
    }
    if (usesTaskShader && !impl_->capabilities.taskShader) {
        return makeError(Error::Unsupported);
    }
    if (desc.usesBindlessHeap && !impl_->capabilities.bindlessDescriptorHeap) {
        return makeError(Error::Unsupported);
    }
    const bool hasDepthStencilFormat = desc.depthStencilFormat != Format::Unknown;
    if ((desc.depthStencil.depthTestEnable || desc.depthStencil.depthWriteEnable) && !hasDepthStencilFormat) {
        return makeError(Error::InvalidArgument);
    }

    detail::PipelineCacheImpl* pipelineCache =
        desc.pipelineCache != nullptr ? desc.pipelineCache->impl_.get() : nullptr;
    if (desc.pipelineCache != nullptr &&
        (pipelineCache == nullptr ||
         pipelineCache->device != impl_.get() ||
         pipelineCache->pipelineCache == VK_NULL_HANDLE)) {
        return makeError(Error::InvalidArgument);
    }
    const uint64_t psoHash = detail::graphicsPipelineStateHash(desc);
    std::unique_lock<std::mutex> pipelineCacheLock;
    if (pipelineCache != nullptr) {
        pipelineCacheLock = std::unique_lock<std::mutex>(pipelineCache->mutex);
    }

    const char* vertexEntryPoint = desc.vertexEntryPoint != nullptr ? desc.vertexEntryPoint : "main";
    const char* taskEntryPoint = desc.taskEntryPoint != nullptr ? desc.taskEntryPoint : "main";
    const char* meshEntryPoint = desc.meshEntryPoint != nullptr ? desc.meshEntryPoint : "main";
    const char* fragmentEntryPoint = desc.fragmentEntryPoint != nullptr ? desc.fragmentEntryPoint : "main";
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(usesTaskShader ? 3u : 2u);
    VkShaderStageFlags graphicsShaderStages = VK_SHADER_STAGE_FRAGMENT_BIT;
    if (usesMeshShader) {
#ifdef VK_EXT_mesh_shader
        if (usesTaskShader) {
            stages.push_back(VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
                .module = desc.taskShader->impl_->module,
                .pName = taskEntryPoint,
            });
            graphicsShaderStages |= VK_SHADER_STAGE_TASK_BIT_EXT;
        }
        stages.push_back(VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
            .module = desc.meshShader->impl_->module,
            .pName = meshEntryPoint,
        });
        graphicsShaderStages |= VK_SHADER_STAGE_MESH_BIT_EXT;
#else
        return makeError(Error::Unsupported);
#endif
    } else {
        stages.push_back(VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = desc.vertexShader->impl_->module,
            .pName = vertexEntryPoint,
        });
        graphicsShaderStages |= VK_SHADER_STAGE_VERTEX_BIT;
    }
    stages.push_back(VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = desc.fragmentShader->impl_->module,
        .pName = fragmentEntryPoint,
    });
    std::array<VkDescriptorSetAndBindingMappingEXT, 3> bindlessMappings{};
    VkShaderDescriptorSetAndBindingMappingInfoEXT bindlessMappingInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
    };
    if (desc.usesBindlessHeap) {
        auto makeHeapMapping = [](uint32_t binding, VkSpirvResourceTypeFlagsEXT resourceMask, uint32_t stride) {
            VkDescriptorSetAndBindingMappingEXT mapping{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .descriptorSet = 0,
                .firstBinding = binding,
                .bindingCount = 1,
                .resourceMask = resourceMask,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
            };
            mapping.sourceData.constantOffset.heapOffset = 0;
            mapping.sourceData.constantOffset.heapArrayStride = stride;
            mapping.sourceData.constantOffset.samplerHeapOffset = 0;
            mapping.sourceData.constantOffset.samplerHeapArrayStride = stride;
            return mapping;
        };

        bindlessMappings[0] = makeHeapMapping(
            0,
            VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT,
            static_cast<uint32_t>(impl_->descriptorHeapWriter.samplerDescriptorSize()));
        bindlessMappings[1] = makeHeapMapping(
            2,
            VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT |
                VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT |
                VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT,
            static_cast<uint32_t>(impl_->descriptorHeapWriter.imageDescriptorSize()));
        bindlessMappings[2] = makeHeapMapping(
            2,
            VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT |
                VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
                VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT,
            static_cast<uint32_t>(impl_->descriptorHeapWriter.bufferDescriptorSize()));
        bindlessMappingInfo.mappingCount = static_cast<uint32_t>(bindlessMappings.size());
        bindlessMappingInfo.pMappings = bindlessMappings.data();
        for (VkPipelineShaderStageCreateInfo& stage : stages) {
            stage.pNext = &bindlessMappingInfo;
        }
    }

    VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = toVkPrimitiveTopology(desc.topology),
    };
    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    VkPipelineRasterizationStateCreateInfo rasterizationState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = toVkCullMode(desc.rasterization.cullMode),
        .frontFace = toVkFrontFace(desc.rasterization.frontFace),
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo multisampleState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineDepthStencilStateCreateInfo depthStencilState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = desc.depthStencil.depthTestEnable ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = desc.depthStencil.depthWriteEnable ? VK_TRUE : VK_FALSE,
        .depthCompareOp = toVkCompareOp(desc.depthStencil.depthCompareOp),
    };
    VkPipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT,
    };
    VkPipelineColorBlendStateCreateInfo colorBlendState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };
    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    VkPushConstantRange bindlessPushConstantRange{
        .stageFlags = graphicsShaderStages,
        .offset = 0,
        .size = sizeof(BindlessHeapPushConstants),
    };
    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = desc.usesBindlessHeap ? 1u : 0u,
        .pPushConstantRanges = desc.usesBindlessHeap ? &bindlessPushConstantRange : nullptr,
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkResult result = VK_SUCCESS;
    if (!desc.usesBindlessHeap) {
        result = vkCreatePipelineLayout(impl_->device, &layoutInfo, nullptr, &layout);
        if (result != VK_SUCCESS) {
            return resultFromVk(result);
        }
    }

    const VkFormat colorFormat = toVkFormat(desc.colorFormat);
    const VkFormat depthStencilFormat = toVkFormat(desc.depthStencilFormat);
    VkPipelineRenderingCreateInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &colorFormat,
        .depthAttachmentFormat = depthStencilFormat,
    };
    VkPipelineCreateFlags2CreateInfo bindlessPipelineFlags{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .pNext = &renderingInfo,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
    };
    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = desc.usesBindlessHeap ? static_cast<const void*>(&bindlessPipelineFlags) :
            static_cast<const void*>(&renderingInfo),
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = usesMeshShader ? nullptr : &vertexInput,
        .pInputAssemblyState = usesMeshShader ? nullptr : &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizationState,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = hasDepthStencilFormat ? &depthStencilState : nullptr,
        .pColorBlendState = &colorBlendState,
        .pDynamicState = &dynamicState,
        .layout = layout,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vkCreateGraphicsPipelines(
        impl_->device,
        pipelineCache != nullptr ? pipelineCache->pipelineCache : VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &pipeline);
    if (result != VK_SUCCESS) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(impl_->device, layout, nullptr);
        }
        return resultFromVk(result);
    }

    auto pipelineImpl = std::make_unique<detail::GraphicsPipelineImpl>();
    pipelineImpl->device = impl_.get();
    pipelineImpl->layout = layout;
    pipelineImpl->pipeline = pipeline;
    pipelineImpl->bindlessPushStages = graphicsShaderStages;
    pipelineImpl->usesBindlessHeap = desc.usesBindlessHeap;
    pipelineImpl->psoHash = psoHash;
    pipelineImpl->pipelineCacheHit = pipelineCache != nullptr &&
        pipelineCache->recordPsoLocked(psoHash);
    outGraphicsPipeline.reset(new GraphicsPipeline(std::move(pipelineImpl)));
    return {};
}

Result Device::createComputePipeline(
    const ComputePipelineDesc& desc,
    std::unique_ptr<ComputePipeline>& outComputePipeline)
{
    outComputePipeline.reset();
    if (impl_ == nullptr ||
        desc.computeShader == nullptr ||
        desc.computeShader->impl_ == nullptr ||
        (desc.bindingMappingCount > 0 && desc.bindingMappings == nullptr) ||
        (!desc.usesBindlessHeap && desc.bindingMappingCount > 0)) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);
    const uint32_t bindlessUserDataOffset = desc.bindingMappingCount == 0
        ? static_cast<uint32_t>(sizeof(BindlessHeapPushConstants))
        : 0u;
    if (desc.usesBindlessHeap) {
        if (!impl_->capabilities.bindlessDescriptorHeap) {
            return makeError(Error::Unsupported);
        }
        const VkDeviceSize requiredPushDataSize =
            bindlessUserDataOffset + desc.bindlessUserPushDataSize;
        if (impl_->descriptorHeapWriter.maxPushDataSize() < requiredPushDataSize) {
            return makeError(Error::Unsupported);
        }
    }

    detail::PipelineCacheImpl* pipelineCache =
        desc.pipelineCache != nullptr ? desc.pipelineCache->impl_.get() : nullptr;
    if (desc.pipelineCache != nullptr &&
        (pipelineCache == nullptr ||
         pipelineCache->device != impl_.get() ||
         pipelineCache->pipelineCache == VK_NULL_HANDLE)) {
        return makeError(Error::InvalidArgument);
    }
    const uint64_t psoHash = detail::computePipelineStateHash(desc);
    std::unique_lock<std::mutex> pipelineCacheLock;
    if (pipelineCache != nullptr) {
        pipelineCacheLock = std::unique_lock<std::mutex>(pipelineCache->mutex);
    }

    const char* computeEntryPoint = desc.computeEntryPoint != nullptr ? desc.computeEntryPoint : "main";
    VkPipelineShaderStageCreateInfo stage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = desc.computeShader->impl_->module,
        .pName = computeEntryPoint,
    };

    std::vector<VkDescriptorSetAndBindingMappingEXT> bindlessMappings;
    VkShaderDescriptorSetAndBindingMappingInfoEXT bindlessMappingInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
    };
    if (desc.usesBindlessHeap) {
        auto makeHeapMapping = [](uint32_t binding, VkSpirvResourceTypeFlagsEXT resourceMask, uint32_t stride) {
            VkDescriptorSetAndBindingMappingEXT mapping{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .descriptorSet = 0,
                .firstBinding = binding,
                .bindingCount = 1,
                .resourceMask = resourceMask,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
            };
            mapping.sourceData.constantOffset.heapOffset = 0;
            mapping.sourceData.constantOffset.heapArrayStride = stride;
            mapping.sourceData.constantOffset.samplerHeapOffset = 0;
            mapping.sourceData.constantOffset.samplerHeapArrayStride = stride;
            return mapping;
        };

        if (desc.bindingMappingCount == 0) {
            bindlessMappings.resize(3);
            bindlessMappings[0] = makeHeapMapping(
                0,
                VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT,
                static_cast<uint32_t>(impl_->descriptorHeapWriter.samplerDescriptorSize()));
            bindlessMappings[1] = makeHeapMapping(
                2,
                VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT |
                    VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT |
                    VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT,
                static_cast<uint32_t>(impl_->descriptorHeapWriter.imageDescriptorSize()));
            bindlessMappings[2] = makeHeapMapping(
                2,
                VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT |
                    VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
                    VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT,
                static_cast<uint32_t>(impl_->descriptorHeapWriter.bufferDescriptorSize()));
        } else {
            bindlessMappings.reserve(desc.bindingMappingCount);
            for (uint32_t index = 0; index < desc.bindingMappingCount; ++index) {
                const ShaderBindingMappingDesc& source = desc.bindingMappings[index];
                if (source.bindingCount == 0) {
                    return makeError(Error::InvalidArgument);
                }

                VkSpirvResourceTypeFlagsEXT resourceMask = 0;
                uint32_t descriptorStride = 0;
                switch (source.type) {
                case ShaderBindingType::Sampler:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.samplerDescriptorSize());
                    break;
                case ShaderBindingType::SampledImage:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.imageDescriptorSize());
                    break;
                case ShaderBindingType::StorageImage:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT |
                        VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.imageDescriptorSize());
                    break;
                case ShaderBindingType::ConstantBuffer:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.bufferDescriptorSize());
                    break;
                case ShaderBindingType::StorageBuffer:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
                        VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.bufferDescriptorSize());
                    break;
                case ShaderBindingType::AccelerationStructure:
                case ShaderBindingType::PartitionedAccelerationStructure:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.bufferDescriptorSize());
                    break;
                }

                const uint32_t valueSize = source.source == ShaderBindingSource::HeapConstantOffset
                    ? 0u
                    : source.source == ShaderBindingSource::DeviceAddressFromPushData
                        ? static_cast<uint32_t>(sizeof(uint64_t))
                        : static_cast<uint32_t>(sizeof(uint32_t));
                if ((valueSize != 0 &&
                     (source.pushDataOffset > desc.bindlessUserPushDataSize ||
                      valueSize > desc.bindlessUserPushDataSize - source.pushDataOffset)) ||
                    (source.source == ShaderBindingSource::DeviceAddressFromPushData &&
                     source.type != ShaderBindingType::ConstantBuffer &&
                     source.type != ShaderBindingType::StorageBuffer &&
                     source.type != ShaderBindingType::AccelerationStructure &&
                     source.type != ShaderBindingType::PartitionedAccelerationStructure)) {
                    return makeError(Error::InvalidArgument);
                }

                VkDescriptorSetAndBindingMappingEXT mapping{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                    .descriptorSet = source.descriptorSet,
                    .firstBinding = source.firstBinding,
                    .bindingCount = source.bindingCount,
                    .resourceMask = resourceMask,
                };
                const uint32_t pushOffset = bindlessUserDataOffset + source.pushDataOffset;
                const uint64_t heapOffset =
                    static_cast<uint64_t>(source.heapIndexOffset) * descriptorStride;
                if (heapOffset > UINT32_MAX) {
                    return makeError(Error::InvalidArgument);
                }
                if (source.source == ShaderBindingSource::DeviceAddressFromPushData) {
                    if (source.heapIndexOffset != 0) {
                        return makeError(Error::InvalidArgument);
                    }
                    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
                    mapping.sourceData.pushAddressOffset = pushOffset;
                } else if (source.source == ShaderBindingSource::HeapConstantOffset) {
                    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
                    mapping.sourceData.constantOffset.heapOffset = static_cast<uint32_t>(heapOffset);
                    mapping.sourceData.constantOffset.heapArrayStride = descriptorStride;
                    mapping.sourceData.constantOffset.samplerHeapOffset = static_cast<uint32_t>(heapOffset);
                    mapping.sourceData.constantOffset.samplerHeapArrayStride = descriptorStride;
                } else {
                    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
                    mapping.sourceData.pushIndex.heapOffset = static_cast<uint32_t>(heapOffset);
                    mapping.sourceData.pushIndex.pushOffset = pushOffset;
                    mapping.sourceData.pushIndex.heapIndexStride = descriptorStride;
                    mapping.sourceData.pushIndex.heapArrayStride = descriptorStride;
                    mapping.sourceData.pushIndex.samplerHeapOffset = static_cast<uint32_t>(heapOffset);
                    mapping.sourceData.pushIndex.samplerPushOffset = pushOffset;
                    mapping.sourceData.pushIndex.samplerHeapIndexStride = descriptorStride;
                    mapping.sourceData.pushIndex.samplerHeapArrayStride = descriptorStride;
                }
                bindlessMappings.push_back(mapping);
            }
        }
        bindlessMappingInfo.mappingCount = static_cast<uint32_t>(bindlessMappings.size());
        bindlessMappingInfo.pMappings = bindlessMappings.data();
        stage.pNext = &bindlessMappingInfo;
    }

    VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkResult result = VK_SUCCESS;
    if (!desc.usesBindlessHeap) {
        result = vkCreatePipelineLayout(impl_->device, &layoutInfo, nullptr, &layout);
        if (result != VK_SUCCESS) {
            return resultFromVk(result);
        }
    }

    VkPipelineCreateFlags2CreateInfo bindlessPipelineFlags{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
    };
    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = desc.usesBindlessHeap ? static_cast<const void*>(&bindlessPipelineFlags) : nullptr,
        .stage = stage,
        .layout = layout,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vkCreateComputePipelines(
        impl_->device,
        pipelineCache != nullptr ? pipelineCache->pipelineCache : VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &pipeline);
    if (result != VK_SUCCESS) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(impl_->device, layout, nullptr);
        }
        return resultFromVk(result);
    }

    auto pipelineImpl = std::make_unique<detail::ComputePipelineImpl>();
    pipelineImpl->device = impl_.get();
    pipelineImpl->layout = layout;
    pipelineImpl->pipeline = pipeline;
    pipelineImpl->usesBindlessHeap = desc.usesBindlessHeap;
    pipelineImpl->bindlessUserDataOffset = bindlessUserDataOffset;
    pipelineImpl->psoHash = psoHash;
    pipelineImpl->pipelineCacheHit = pipelineCache != nullptr &&
        pipelineCache->recordPsoLocked(psoHash);
    outComputePipeline.reset(new ComputePipeline(std::move(pipelineImpl)));
    return {};
}

Result Device::createGraphicsShaderObjectProgram(
    const GraphicsShaderObjectProgramDesc& desc,
    std::unique_ptr<GraphicsShaderObjectProgram>& outProgram)
{
    outProgram.reset();
    if (impl_ == nullptr ||
        desc.vertexCode == nullptr ||
        desc.vertexByteSize == 0 ||
        (desc.vertexByteSize % sizeof(uint32_t)) != 0 ||
        desc.fragmentCode == nullptr ||
        desc.fragmentByteSize == 0 ||
        (desc.fragmentByteSize % sizeof(uint32_t)) != 0) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);
    if (!impl_->capabilities.shaderObject || !impl_->shaderObjectEnabled) {
        return makeError(Error::Unsupported);
    }
    if (desc.usesBindlessHeap) {
        if (!impl_->capabilities.bindlessDescriptorHeap) {
            return makeError(Error::Unsupported);
        }
        const VkDeviceSize requiredPushDataSize =
            sizeof(BindlessHeapPushConstants) + desc.bindlessUserPushDataSize;
        if (impl_->descriptorHeapWriter.maxPushDataSize() < requiredPushDataSize) {
            return makeError(Error::Unsupported);
        }
    }

    std::array<VkDescriptorSetAndBindingMappingEXT, 3> bindlessMappings{};
    VkShaderDescriptorSetAndBindingMappingInfoEXT bindlessMappingInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
    };
    if (desc.usesBindlessHeap) {
        auto makeHeapMapping = [](uint32_t binding, VkSpirvResourceTypeFlagsEXT resourceMask, uint32_t stride) {
            VkDescriptorSetAndBindingMappingEXT mapping{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                .descriptorSet = 0,
                .firstBinding = binding,
                .bindingCount = 1,
                .resourceMask = resourceMask,
                .source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT,
            };
            mapping.sourceData.constantOffset.heapOffset = 0;
            mapping.sourceData.constantOffset.heapArrayStride = stride;
            mapping.sourceData.constantOffset.samplerHeapOffset = 0;
            mapping.sourceData.constantOffset.samplerHeapArrayStride = stride;
            return mapping;
        };

        bindlessMappings[0] = makeHeapMapping(
            0,
            VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT,
            static_cast<uint32_t>(impl_->descriptorHeapWriter.samplerDescriptorSize()));
        bindlessMappings[1] = makeHeapMapping(
            2,
            VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT |
                VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT |
                VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT,
            static_cast<uint32_t>(impl_->descriptorHeapWriter.imageDescriptorSize()));
        bindlessMappings[2] = makeHeapMapping(
            2,
            VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT |
                VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
                VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT,
            static_cast<uint32_t>(impl_->descriptorHeapWriter.bufferDescriptorSize()));
        bindlessMappingInfo.mappingCount = static_cast<uint32_t>(bindlessMappings.size());
        bindlessMappingInfo.pMappings = bindlessMappings.data();
    }

    const char* vertexEntryPoint = desc.vertexEntryPoint != nullptr ? desc.vertexEntryPoint : "main";
    const char* fragmentEntryPoint = desc.fragmentEntryPoint != nullptr ? desc.fragmentEntryPoint : "main";
    const VkShaderCreateFlagsEXT shaderFlags =
        VK_SHADER_CREATE_LINK_STAGE_BIT_EXT |
        (desc.usesBindlessHeap ? VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT : 0);
    std::array<VkShaderCreateInfoEXT, 2> shaderInfos{
        VkShaderCreateInfoEXT{
            .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
            .pNext = desc.usesBindlessHeap ? static_cast<const void*>(&bindlessMappingInfo) : nullptr,
            .flags = shaderFlags,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
            .codeSize = static_cast<size_t>(desc.vertexByteSize),
            .pCode = desc.vertexCode,
            .pName = vertexEntryPoint,
        },
        VkShaderCreateInfoEXT{
            .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
            .pNext = desc.usesBindlessHeap ? static_cast<const void*>(&bindlessMappingInfo) : nullptr,
            .flags = shaderFlags,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .nextStage = 0,
            .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
            .codeSize = static_cast<size_t>(desc.fragmentByteSize),
            .pCode = desc.fragmentCode,
            .pName = fragmentEntryPoint,
        },
    };

    std::array<VkShaderEXT, 2> shaders{
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
    };
    const VkResult result = vkCreateShadersEXT(
        impl_->device,
        static_cast<uint32_t>(shaderInfos.size()),
        shaderInfos.data(),
        nullptr,
        shaders.data());
    if (result != VK_SUCCESS) {
        for (VkShaderEXT shader : shaders) {
            if (shader != VK_NULL_HANDLE) {
                vkDestroyShaderEXT(impl_->device, shader, nullptr);
            }
        }
        return resultFromVk(result);
    }
    profiling::registerNsightAftermathShaderBinary(desc.vertexCode, desc.vertexByteSize);
    profiling::registerNsightAftermathShaderBinary(desc.fragmentCode, desc.fragmentByteSize);

    auto programImpl = std::make_unique<detail::GraphicsShaderObjectProgramImpl>();
    programImpl->device = impl_.get();
    programImpl->vertexShader = shaders[0];
    programImpl->fragmentShader = shaders[1];
    programImpl->usesBindlessHeap = desc.usesBindlessHeap;
    outProgram.reset(new GraphicsShaderObjectProgram(std::move(programImpl)));
    return {};
}

Result Device::createBindlessHeap(const BindlessHeapDesc& desc, std::unique_ptr<BindlessHeap>& outBindlessHeap)
{
    outBindlessHeap.reset();
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);
    if (!impl_->capabilities.bindlessDescriptorHeap) {
        return makeError(Error::Unsupported);
    }

    auto bindlessImpl = std::make_unique<detail::BindlessHeapImpl>();
    Result result = bindlessImpl->initialize(*impl_, desc);
    if (!result) {
        return result;
    }

    outBindlessHeap.reset(new BindlessHeap(std::move(bindlessImpl)));
    return {};
}

Result createDevice(const DeviceDesc& desc, std::unique_ptr<Device>& outDevice)
{
    outDevice.reset();

    auto deviceImpl = std::make_unique<detail::DeviceImpl>();
    if (desc.enableAftermath && profiling::nsightAftermathSdkAvailable()) {
        profiling::initializeNsightAftermath(desc.applicationName);
    }

    PFN_vkGetInstanceProcAddr streamlineVkGetInstanceProcAddr = nullptr;
    if (desc.enableStreamline && vulkan::streamlineSdkAvailable()) {
        const char* const vulkanLibraryName = vulkan::streamlineVulkanLibraryName();
        streamlineVkGetInstanceProcAddr =
            loadVulkanLoaderProcAddr(vulkanLibraryName, deviceImpl->vulkanLoaderHandle);
        if (streamlineVkGetInstanceProcAddr != nullptr) {
            std::string streamlineLog;
            Result streamlineResult = vulkan::initializeStreamlinePreDevice(streamlineLog);
            if (streamlineResult) {
                deviceImpl->streamlineInitialized = true;
            } else {
                spdlog::warn(
                    "NVIDIA Streamline initialization skipped: {}",
                    streamlineLog.empty() ? resultToString(streamlineResult) : streamlineLog);
                vulkan::shutdownStreamline();
                SDL_UnloadObject(deviceImpl->vulkanLoaderHandle);
                deviceImpl->vulkanLoaderHandle = nullptr;
                streamlineVkGetInstanceProcAddr = nullptr;
            }
        } else {
            spdlog::warn(
                "SDL_LoadObject({}) failed: {}; retrying without Streamline.",
                vulkanLibraryName,
                SDL_GetError());
        }
    }

    if (streamlineVkGetInstanceProcAddr == nullptr) {
        if (!acquireSdlVulkanLibrary()) {
            spdlog::error("SDL_Vulkan_LoadLibrary failed: {}", SDL_GetError());
            return makeError(Error::Unsupported);
        }
        deviceImpl->sdlVulkanLoaded = true;
    }

    VkResult vkResult = VK_SUCCESS;
    if (streamlineVkGetInstanceProcAddr != nullptr) {
        volkInitializeCustom(streamlineVkGetInstanceProcAddr);
    } else {
        vkResult = volkInitialize();
        if (vkResult != VK_SUCCESS) {
            spdlog::error("volkInitialize failed with VkResult {}", static_cast<int>(vkResult));
            return resultFromVk(vkResult);
        }
    }

    Uint32 sdlExtensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    if (sdlExtensions == nullptr || sdlExtensionCount == 0) {
        spdlog::error("SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError());
        return makeError(Error::Unsupported);
    }

    std::vector<const char*> instanceExtensions;
    instanceExtensions.reserve(sdlExtensionCount + 1);
    for (Uint32 index = 0; index < sdlExtensionCount; ++index) {
        instanceExtensions.push_back(sdlExtensions[index]);
    }

    const std::vector<VkExtensionProperties> availableExtensions = enumerateInstanceExtensions();
    const bool debugUtilsAvailable = hasName(availableExtensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugUtilsAvailable) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        deviceImpl->debugUtilsEnabled = true;
    }
    // Required by the NRC SDK's physical-device feature queries.
    if (hasName(availableExtensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }

    std::vector<const char*> instanceLayers;
    const std::vector<VkLayerProperties> availableLayers = enumerateInstanceLayers();
    if (desc.enableValidation && hasName(availableLayers, "VK_LAYER_KHRONOS_validation")) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
        deviceImpl->validationEnabled = true;
    } else if (desc.enableValidation) {
        spdlog::warn("Vulkan validation requested but VK_LAYER_KHRONOS_validation is not available.");
    }

    VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = desc.applicationName,
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Metallic",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = kVulkanApiVersion,
    };

    VkInstanceCreateInfo instanceInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &applicationInfo,
        .enabledLayerCount = static_cast<uint32_t>(instanceLayers.size()),
        .ppEnabledLayerNames = instanceLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
        .ppEnabledExtensionNames = instanceExtensions.data(),
    };

    vkResult = vkCreateInstance(&instanceInfo, nullptr, &deviceImpl->instance);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }
    volkLoadInstance(deviceImpl->instance);

    if (deviceImpl->validationEnabled && debugUtilsAvailable) {
        deviceImpl->debugMessenger = createDebugMessenger(deviceImpl->instance);
    }

    uint32_t physicalDeviceCount = 0;
    vkResult = vkEnumeratePhysicalDevices(deviceImpl->instance, &physicalDeviceCount, nullptr);
    if (vkResult != VK_SUCCESS || physicalDeviceCount == 0) {
        if (vkResult == VK_SUCCESS) {
            return makeError(Error::Unsupported);
        }
        return resultFromVk(vkResult);
    }

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(deviceImpl->instance, &physicalDeviceCount, physicalDevices.data());

    const VulkanDeviceFeatureRequest requestedFeatures = VulkanDeviceFeatureRequest::from(desc);
    VulkanPhysicalDeviceCandidate bestCandidate;
    VulkanDeviceFeatureSelection selectedFeatures;

    for (VkPhysicalDevice physicalDevice : physicalDevices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        if (properties.apiVersion < kVulkanApiVersion) {
            continue;
        }

        const VulkanExtensionSet extensions = VulkanExtensionSet::query(physicalDevice);
        if (!extensions.swapchain) {
            continue;
        }

        VulkanDeviceFeatureProbe probe;
        probe.query(physicalDevice, extensions);
        if (!probe.supportsRequiredCoreFeatures()) {
            continue;
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        uint32_t graphicsFamily = UINT32_MAX;
        uint32_t computeFamily = UINT32_MAX;
        for (uint32_t queueIndex = 0; queueIndex < queueFamilyCount; ++queueIndex) {
            const VkQueueFlags queueFlags = queueFamilies[queueIndex].queueFlags;
            if ((queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                if (graphicsFamily == UINT32_MAX) {
                    graphicsFamily = queueIndex;
                }
                if ((queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    graphicsFamily = queueIndex;
                    computeFamily = queueIndex;
                    break;
                }
            }
        }

        if (computeFamily == UINT32_MAX) {
            for (uint32_t queueIndex = 0; queueIndex < queueFamilyCount; ++queueIndex) {
                if ((queueFamilies[queueIndex].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    computeFamily = queueIndex;
                    break;
                }
            }
        }
        if (graphicsFamily == UINT32_MAX || computeFamily == UINT32_MAX) {
            continue;
        }

        uint32_t copyFamily = UINT32_MAX;
        for (uint32_t queueIndex = 0; queueIndex < queueFamilyCount; ++queueIndex) {
            const VkQueueFlags flags = queueFamilies[queueIndex].queueFlags;
            if ((flags & VK_QUEUE_TRANSFER_BIT) != 0 &&
                (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0) {
                copyFamily = queueIndex;
                break;
            }
        }
        if (copyFamily == UINT32_MAX) {
            for (uint32_t queueIndex = 0; queueIndex < queueFamilyCount; ++queueIndex) {
                const VkQueueFlags flags = queueFamilies[queueIndex].queueFlags;
                if ((flags & VK_QUEUE_TRANSFER_BIT) != 0 && (flags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                    copyFamily = queueIndex;
                    break;
                }
            }
        }
        if (copyFamily == UINT32_MAX) {
            for (uint32_t queueIndex = 0; queueIndex < queueFamilyCount; ++queueIndex) {
                if ((queueFamilies[queueIndex].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0) {
                    copyFamily = queueIndex;
                    break;
                }
            }
        }

        const VulkanDeviceFeatureSelection featureSelection = VulkanDeviceFeatureSelection::select(
            requestedFeatures,
            physicalDevice,
            extensions,
            probe);
        const int32_t featureScore = featureSelection.score();
        if (featureScore > bestCandidate.featureScore) {
            bestCandidate = VulkanPhysicalDeviceCandidate{
                .physicalDevice = physicalDevice,
                .graphicsFamily = graphicsFamily,
                .computeFamily = computeFamily,
                .copyFamily = copyFamily,
                .features = featureSelection,
                .featureScore = featureScore,
            };
        }
        if (featureSelection.matches(requestedFeatures) &&
            (!requestedFeatures.streamline || featureSelection.streamline)) {
            deviceImpl->physicalDevice = physicalDevice;
            deviceImpl->graphicsFamily = graphicsFamily;
            deviceImpl->computeFamily = computeFamily;
            deviceImpl->copyFamily = copyFamily;
            selectedFeatures = featureSelection;
            break;
        }
    }

    if (deviceImpl->physicalDevice == VK_NULL_HANDLE && bestCandidate.physicalDevice != VK_NULL_HANDLE) {
        deviceImpl->physicalDevice = bestCandidate.physicalDevice;
        deviceImpl->graphicsFamily = bestCandidate.graphicsFamily;
        deviceImpl->computeFamily = bestCandidate.computeFamily;
        deviceImpl->copyFamily = bestCandidate.copyFamily;
        selectedFeatures = bestCandidate.features;
    }

    if (deviceImpl->physicalDevice == VK_NULL_HANDLE) {
        return makeError(Error::Unsupported);
    }

    uint32_t selectedQueueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        deviceImpl->physicalDevice,
        &selectedQueueFamilyCount,
        nullptr);
    std::vector<VkQueueFamilyProperties> selectedQueueFamilies(selectedQueueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        deviceImpl->physicalDevice,
        &selectedQueueFamilyCount,
        selectedQueueFamilies.data());

    deviceImpl->copyQueueIndex = UINT32_MAX;
    if (deviceImpl->copyFamily < selectedQueueFamilies.size()) {
        const bool copyUsesExistingQueue =
            deviceImpl->copyFamily == deviceImpl->graphicsFamily ||
            deviceImpl->copyFamily == deviceImpl->computeFamily;
        if (!copyUsesExistingQueue) {
            deviceImpl->copyQueueIndex = 0;
        } else if (selectedQueueFamilies[deviceImpl->copyFamily].queueCount > 1) {
            deviceImpl->copyQueueIndex = 1;
        }
    }

    struct QueueFamilyRequest {
        uint32_t family = UINT32_MAX;
        uint32_t count = 0;
    };
    std::vector<QueueFamilyRequest> queueRequests;
    const auto requestQueues = [&queueRequests](uint32_t family, uint32_t count) {
        const auto found = std::find_if(
            queueRequests.begin(),
            queueRequests.end(),
            [family](const QueueFamilyRequest& request) { return request.family == family; });
        if (found == queueRequests.end()) {
            queueRequests.push_back(QueueFamilyRequest{.family = family, .count = count});
        } else {
            found->count = std::max(found->count, count);
        }
    };
    requestQueues(deviceImpl->graphicsFamily, 1);
    requestQueues(deviceImpl->computeFamily, 1);
    if (deviceImpl->copyQueueIndex != UINT32_MAX) {
        requestQueues(deviceImpl->copyFamily, deviceImpl->copyQueueIndex + 1);
    }

    const std::array<float, 2> queuePriorities{1.0f, 1.0f};
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(queueRequests.size());
    for (const QueueFamilyRequest& request : queueRequests) {
        queueInfos.push_back({
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = request.family,
            .queueCount = request.count,
            .pQueuePriorities = queuePriorities.data(),
        });
    }

    VulkanEnabledFeatureChain enabledFeatureChain(selectedFeatures);
    std::vector<const char*> deviceExtensions = enabledDeviceExtensions(selectedFeatures);
    VkDeviceCreateInfo deviceInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &enabledFeatureChain.features,
        .queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size()),
        .pQueueCreateInfos = queueInfos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };

    vkResult = vkCreateDevice(deviceImpl->physicalDevice, &deviceInfo, nullptr, &deviceImpl->device);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }
    activateVolkDevice(deviceImpl->device);

    VkPhysicalDeviceProperties selectedProperties{};
    vkGetPhysicalDeviceProperties(deviceImpl->physicalDevice, &selectedProperties);
    deviceImpl->pipelineCacheFileIdentity.backendTag = kVulkanPipelineCacheBackendTag;
    std::memcpy(
        deviceImpl->pipelineCacheFileIdentity.compatibilityKey.data() + 0,
        &selectedProperties.vendorID,
        sizeof(selectedProperties.vendorID));
    std::memcpy(
        deviceImpl->pipelineCacheFileIdentity.compatibilityKey.data() + 4,
        &selectedProperties.deviceID,
        sizeof(selectedProperties.deviceID));
    std::memcpy(
        deviceImpl->pipelineCacheFileIdentity.compatibilityKey.data() + 8,
        &selectedProperties.driverVersion,
        sizeof(selectedProperties.driverVersion));
    std::memcpy(
        deviceImpl->pipelineCacheFileIdentity.compatibilityKey.data() + 12,
        &selectedProperties.apiVersion,
        sizeof(selectedProperties.apiVersion));
    std::memcpy(
        deviceImpl->pipelineCacheFileIdentity.compatibilityKey.data() + 16,
        selectedProperties.pipelineCacheUUID,
        VK_UUID_SIZE);
    deviceImpl->capabilities.timestampPeriodNanoseconds =
        static_cast<double>(selectedProperties.limits.timestampPeriod);
    deviceImpl->capabilities.timestampQueries =
        deviceImpl->graphicsFamily < selectedQueueFamilies.size() &&
        selectedQueueFamilies[deviceImpl->graphicsFamily].timestampValidBits != 0 &&
        deviceImpl->capabilities.timestampPeriodNanoseconds > 0.0;
    deviceImpl->capabilities.bufferCopyOffsetAlignment =
        std::max<uint64_t>(selectedProperties.limits.optimalBufferCopyOffsetAlignment, 1);
    deviceImpl->capabilities.textureUploadBufferOffsetAlignment =
        std::max<uint64_t>(selectedProperties.limits.optimalBufferCopyOffsetAlignment, 1);
    deviceImpl->capabilities.textureUploadRowPitchAlignment =
        std::max<uint64_t>(selectedProperties.limits.optimalBufferCopyRowPitchAlignment, 1);
    deviceImpl->capabilities.textureUploadSlicePitchAlignment =
        std::max<uint64_t>(selectedProperties.limits.optimalBufferCopyRowPitchAlignment, 1);
    deviceImpl->capabilities.constantBufferOffsetAlignment =
        std::max<uint64_t>(selectedProperties.limits.minUniformBufferOffsetAlignment, 1);

    if (selectedFeatures.bindlessDescriptorHeap) {
        vkResult = deviceImpl->descriptorHeapWriter.initialize(deviceImpl->physicalDevice, deviceImpl->device);
        if (vkResult != VK_SUCCESS) {
            return resultFromVk(vkResult);
        }

        const VkDeviceSize samplerCapacityBytes =
            deviceImpl->descriptorHeapWriter.maxSamplerHeapSize() >
                deviceImpl->descriptorHeapWriter.minSamplerHeapReservedRange()
            ? deviceImpl->descriptorHeapWriter.maxSamplerHeapSize() -
                deviceImpl->descriptorHeapWriter.minSamplerHeapReservedRange()
            : 0;
        const VkDeviceSize resourceCapacityBytes =
            deviceImpl->descriptorHeapWriter.maxResourceHeapSize() >
                deviceImpl->descriptorHeapWriter.minResourceHeapReservedRange()
            ? deviceImpl->descriptorHeapWriter.maxResourceHeapSize() -
                deviceImpl->descriptorHeapWriter.minResourceHeapReservedRange()
            : 0;

        deviceImpl->capabilities.bindlessDescriptorHeap = true;
        deviceImpl->capabilities.maxBindlessSamplers = capacityFromBytes(
            samplerCapacityBytes,
            deviceImpl->descriptorHeapWriter.samplerDescriptorSize());
        deviceImpl->capabilities.maxBindlessSampledImages = capacityFromBytes(
            resourceCapacityBytes,
            deviceImpl->descriptorHeapWriter.imageDescriptorSize());
        deviceImpl->capabilities.maxBindlessBuffers = capacityFromBytes(
            resourceCapacityBytes,
            deviceImpl->descriptorHeapWriter.bufferDescriptorSize());
        deviceImpl->bindlessDescriptorHeapEnabled = true;
    }
    deviceImpl->capabilities.shaderObject = selectedFeatures.shaderObject;
    deviceImpl->shaderObjectEnabled = selectedFeatures.shaderObject;
    deviceImpl->capabilities.meshShader = selectedFeatures.meshShader;
    deviceImpl->capabilities.taskShader = selectedFeatures.taskShader;
    deviceImpl->capabilities.geometryShader = selectedFeatures.geometryShader;
    deviceImpl->capabilities.rayTracingAccelerationStructure = selectedFeatures.rayTracingAccelerationStructure;
    deviceImpl->rayTracingAccelerationStructureEnabled = selectedFeatures.rayTracingAccelerationStructure;
    deviceImpl->capabilities.rayQuery = selectedFeatures.rayQuery;
    deviceImpl->rayQueryEnabled = selectedFeatures.rayQuery;
    deviceImpl->capabilities.pushDescriptor = selectedFeatures.pushDescriptor;
    deviceImpl->pushDescriptorEnabled = selectedFeatures.pushDescriptor;
    deviceImpl->capabilities.clusterAccelerationStructure = selectedFeatures.clusterAccelerationStructure;
    deviceImpl->clusterAccelerationStructureEnabled = selectedFeatures.clusterAccelerationStructure;
    deviceImpl->capabilities.partitionedAccelerationStructure =
        selectedFeatures.partitionedAccelerationStructure;
    deviceImpl->partitionedAccelerationStructureEnabled =
        selectedFeatures.partitionedAccelerationStructure;
    deviceImpl->capabilities.aftermath = selectedFeatures.aftermath;
    deviceImpl->capabilities.shaderIntegerDotProduct =
        selectedFeatures.shaderIntegerDotProduct;
    deviceImpl->capabilities.cooperativeVector = selectedFeatures.cooperativeVector;
    deviceImpl->bufferDeviceAddressEnabled = selectedFeatures.usesBufferDeviceAddress();

    if (deviceImpl->debugUtilsEnabled) {
        deviceImpl->setDebugUtilsObjectName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(deviceImpl->device, "vkSetDebugUtilsObjectNameEXT"));
        deviceImpl->cmdBeginDebugUtilsLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetDeviceProcAddr(deviceImpl->device, "vkCmdBeginDebugUtilsLabelEXT"));
        deviceImpl->cmdEndDebugUtilsLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetDeviceProcAddr(deviceImpl->device, "vkCmdEndDebugUtilsLabelEXT"));
    }

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = deviceImpl->physicalDevice;
    allocatorInfo.device = deviceImpl->device;
    allocatorInfo.instance = deviceImpl->instance;
    allocatorInfo.vulkanApiVersion = kVulkanApiVersion;
    if (deviceImpl->bufferDeviceAddressEnabled) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    VmaVulkanFunctions vulkanFunctions{};
    vkResult = vmaImportVulkanFunctionsFromVolk(&allocatorInfo, &vulkanFunctions);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;
    vkResult = vmaCreateAllocator(&allocatorInfo, &deviceImpl->allocator);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(deviceImpl->device, deviceImpl->graphicsFamily, 0, &graphicsQueue);
    deviceImpl->addQueue(
        graphicsQueue,
        deviceImpl->graphicsFamily,
        selectedQueueFamilies[deviceImpl->graphicsFamily].timestampValidBits,
        QueueType::Graphics);

    VkQueue computeQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(deviceImpl->device, deviceImpl->computeFamily, 0, &computeQueue);
    deviceImpl->addQueue(
        computeQueue,
        deviceImpl->computeFamily,
        selectedQueueFamilies[deviceImpl->computeFamily].timestampValidBits,
        QueueType::Compute);

    if (deviceImpl->copyQueueIndex != UINT32_MAX) {
        VkQueue copyQueue = VK_NULL_HANDLE;
        vkGetDeviceQueue(
            deviceImpl->device,
            deviceImpl->copyFamily,
            deviceImpl->copyQueueIndex,
            &copyQueue);
        if (copyQueue != VK_NULL_HANDLE && copyQueue != graphicsQueue && copyQueue != computeQueue) {
            deviceImpl->addQueue(
                copyQueue,
                deviceImpl->copyFamily,
                selectedQueueFamilies[deviceImpl->copyFamily].timestampValidBits,
                QueueType::Copy);
            deviceImpl->capabilities.independentCopyQueue = true;
        } else {
            deviceImpl->copyQueueIndex = UINT32_MAX;
        }
    }

    if (deviceImpl->streamlineInitialized && selectedFeatures.streamline) {
        std::string streamlineLog;
        Result streamlineResult = setStreamlineVulkanDevice(
            vulkan::NativeDevice{
                .instance = deviceImpl->instance,
                .physicalDevice = deviceImpl->physicalDevice,
                .device = deviceImpl->device,
                .apiVersion = kVulkanApiVersion,
            },
            vulkan::NativeQueue{
                .queue = graphicsQueue,
                .familyIndex = deviceImpl->graphicsFamily,
            },
            vulkan::NativeQueue{
                .queue = computeQueue,
                .familyIndex = deviceImpl->computeFamily,
            },
            streamlineLog);
        if (streamlineResult) {
            deviceImpl->capabilities.streamline = true;
            deviceImpl->capabilities.streamlineDlssRr = vulkan::streamlineDlssRrSupported();
            if (!deviceImpl->capabilities.streamlineDlssRr && !streamlineLog.empty()) {
                spdlog::warn("NVIDIA Streamline DLSS-RR unsupported: {}", streamlineLog);
            }
        } else {
            spdlog::error(
                "NVIDIA Streamline Vulkan setup failed: {}",
                streamlineLog.empty() ? resultToString(streamlineResult) : streamlineLog);
        }
    } else if (deviceImpl->streamlineInitialized && desc.enableStreamline) {
        spdlog::warn("NVIDIA Streamline initialized, but the selected Vulkan device is missing required extensions.");
    }
    if (desc.enableAftermath &&
        profiling::nsightAftermathInitialized() &&
        !selectedFeatures.aftermath) {
        spdlog::warn(
            "NVIDIA Nsight Aftermath initialized, but the selected Vulkan device is missing required diagnostics support.");
    }
    if (selectedFeatures.cooperativeVector) {
        spdlog::info("[Vulkan] VK_NV_cooperative_vector enabled");
    }

    outDevice.reset(new Device(std::move(deviceImpl)));
    return {};
}

namespace detail {

struct VulkanNativeAccess {
    static vulkan::NativeDevice nativeDevice(Device& device)
    {
        if (device.impl_ == nullptr) {
            return {};
        }
        return vulkan::NativeDevice{
            .instance = device.impl_->instance,
            .physicalDevice = device.impl_->physicalDevice,
            .device = device.impl_->device,
            .apiVersion = kVulkanApiVersion,
        };
    }

    static vulkan::NativeQueue nativeQueue(Queue& queue)
    {
        if (queue.impl_ == nullptr) {
            return {};
        }
        return vulkan::NativeQueue{
            .queue = queue.impl_->queue,
            .familyIndex = queue.impl_->familyIndex,
        };
    }

    static vulkan::NativeBuffer nativeBuffer(Buffer& buffer)
    {
        if (buffer.impl_ == nullptr) {
            return {};
        }

        VkDeviceAddress address = 0;
        const BufferUsageBits usage = buffer.impl_->desc.usage;
        const bool expectsAddress =
            hasFlag(usage, BufferUsageBits::ShaderDeviceAddress) ||
            hasFlag(usage, BufferUsageBits::AccelerationStructureBuildInput) ||
            hasFlag(usage, BufferUsageBits::AccelerationStructureStorage) ||
            (buffer.impl_->device != nullptr &&
                buffer.impl_->device->bufferDeviceAddressEnabled &&
                (hasFlag(usage, BufferUsageBits::Constant) || hasFlag(usage, BufferUsageBits::Storage)));
        if (expectsAddress &&
            buffer.impl_->device != nullptr &&
            buffer.impl_->buffer != VK_NULL_HANDLE) {
            activateVolkDevice(buffer.impl_->device->device);
            VkBufferDeviceAddressInfo addressInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .buffer = buffer.impl_->buffer,
            };
            address = vkGetBufferDeviceAddress(buffer.impl_->device->device, &addressInfo);
        }

        return vulkan::NativeBuffer{
            .buffer = buffer.impl_->buffer,
            .address = address,
            .size = buffer.impl_->desc.size,
        };
    }

    static vulkan::NativeTexture nativeTexture(Texture& texture)
    {
        if (texture.impl_ == nullptr) {
            return {};
        }

        const TextureDesc& desc = texture.impl_->desc;
        return vulkan::NativeTexture{
            .image = texture.impl_->image,
            .memory = texture.impl_->memory,
            .format = toVkFormat(desc.format),
            .width = desc.width,
            .height = desc.height,
            .depth = desc.depth,
            .mipCount = desc.mipCount,
            .layerCount = desc.layerCount,
            .flags = texture.impl_->flags,
            .usage = texture.impl_->usage,
        };
    }

    static VkCommandBuffer nativeCommandBuffer(CommandBuffer& commandBuffer)
    {
        return commandBuffer.impl_ != nullptr ? commandBuffer.impl_->commandBuffer : VK_NULL_HANDLE;
    }

    static VkFormat nativeSwapchainFormat(Swapchain& swapchain)
    {
        return swapchain.impl_ != nullptr ? swapchain.impl_->vkFormat : VK_FORMAT_UNDEFINED;
    }

    static VkImageView nativeImageView(TextureView& view)
    {
        return view.impl_ != nullptr ? view.impl_->view : VK_NULL_HANDLE;
    }

};

} // namespace detail

namespace vulkan {

NativeDevice nativeDevice(Device& device)
{
    return detail::VulkanNativeAccess::nativeDevice(device);
}

NativeQueue nativeQueue(Queue& queue)
{
    return detail::VulkanNativeAccess::nativeQueue(queue);
}

NativeBuffer nativeBuffer(Buffer& buffer)
{
    return detail::VulkanNativeAccess::nativeBuffer(buffer);
}

NativeTexture nativeTexture(Texture& texture)
{
    return detail::VulkanNativeAccess::nativeTexture(texture);
}

VkCommandBuffer nativeCommandBuffer(CommandBuffer& commandBuffer)
{
    return detail::VulkanNativeAccess::nativeCommandBuffer(commandBuffer);
}

VkFormat nativeSwapchainFormat(Swapchain& swapchain)
{
    return detail::VulkanNativeAccess::nativeSwapchainFormat(swapchain);
}

VkImageView nativeImageView(TextureView& view)
{
    return detail::VulkanNativeAccess::nativeImageView(view);
}

} // namespace vulkan

namespace {

int resultToExitCode(Result result)
{
    return result ? 0 : 1;
}

bool checkResult(Result result, const char* label)
{
    if (result) {
        return true;
    }

    spdlog::error("{} failed with Result {}", label, resultToString(result));
    return false;
}

constexpr const char* kTriangleShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kTriangleShaderModuleName = "Triangle";
constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";
constexpr const char* kBindlessSmokeShaderModuleName = "BindlessSmoke";
constexpr const char* kBindlessSmokeVertexEntryPoint = "bindlessSmokeVertexMain";
constexpr const char* kBindlessSmokeFragmentEntryPoint = "bindlessSmokeFragmentMain";

Result createSlangShaderModule(
    Device& device,
    const char* moduleName,
    const char* entryPointName,
    std::unique_ptr<ShaderModule>& outShaderModule)
{
    ShaderCompileResult compileResult;
    Result result = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kTriangleShaderSearchPath,
        },
        compileResult);
    if (!result) {
        spdlog::error("Slang compile failed for {}.{}", moduleName, entryPointName);
        if (!compileResult.diagnostics.empty()) {
            spdlog::error("{}", compileResult.diagnostics);
        }
        return result;
    }
    if (!compileResult.diagnostics.empty()) {
        spdlog::warn("{}", compileResult.diagnostics);
    }

    const std::string shaderDebugName = std::string(moduleName) + "." + entryPointName;
    return device.createShaderModule(
        ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            .debugName = shaderDebugName.c_str(),
        },
        outShaderModule);
}

Result createTriangleShaderModule(Device& device, const char* entryPointName, std::unique_ptr<ShaderModule>& outShaderModule)
{
    return createSlangShaderModule(device, kTriangleShaderModuleName, entryPointName, outShaderModule);
}

} // namespace

namespace detail {

struct TrianglePreviewRendererImpl {
    std::unique_ptr<Device> device;
    Queue* graphicsQueue = nullptr;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Fence> fence;
    std::unique_ptr<ShaderModule> vertexShader;
    std::unique_ptr<ShaderModule> fragmentShader;
    std::unique_ptr<GraphicsPipeline> pipeline;
    std::unique_ptr<Texture> colorTexture;
    std::unique_ptr<TextureView> colorTextureView;
    std::unique_ptr<Buffer> readbackBuffer;
    std::vector<uint32_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;

    Result initialize(bool enableValidation);
    Result ensureResources(uint32_t newWidth, uint32_t newHeight);
    Result render(uint32_t newWidth, uint32_t newHeight);
};

Result TrianglePreviewRendererImpl::initialize(bool enableValidation)
{
    Result result = createDevice(
        DeviceDesc{
            .applicationName = "Metallic Triangle Preview",
            .enableValidation = enableValidation,
            .enableAftermath = true,
        },
        device);
    if (!result) {
        return result;
    }

    graphicsQueue = device->getQueue(QueueType::Graphics);
    if (graphicsQueue == nullptr) {
        return makeError(Error::Unsupported);
    }

    result = device->createCommandPool(*graphicsQueue, commandPool);
    if (!result) {
        return result;
    }
    result = commandPool->createCommandBuffer(commandBuffer);
    if (!result) {
        return result;
    }
    result = device->createFence(true, fence);
    if (!result) {
        return result;
    }

    result = createTriangleShaderModule(*device, kTriangleVertexEntryPoint, vertexShader);
    if (!result) {
        return result;
    }
    result = createTriangleShaderModule(*device, kTriangleFragmentEntryPoint, fragmentShader);
    if (!result) {
        return result;
    }

    return device->createGraphicsPipeline(
        GraphicsPipelineDesc{
            .vertexShader = vertexShader.get(),
            .fragmentShader = fragmentShader.get(),
            .colorFormat = Format::Rgba8Unorm,
            .topology = PrimitiveTopology::TriangleList,
        },
        pipeline);
}

Result TrianglePreviewRendererImpl::ensureResources(uint32_t newWidth, uint32_t newHeight)
{
    if (newWidth == 0 || newHeight == 0) {
        return makeError(Error::InvalidArgument);
    }

    if (newWidth == width && newHeight == height && colorTexture != nullptr && readbackBuffer != nullptr) {
        return {};
    }

    if (device == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    (void)device->waitIdle();
    colorTextureView.reset();
    colorTexture.reset();
    readbackBuffer.reset();

    Result result = device->createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
            .format = Format::Rgba8Unorm,
            .width = newWidth,
            .height = newHeight,
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        },
        colorTexture);
    if (!result) {
        return result;
    }

    result = device->createTextureView(
        *colorTexture,
        TextureViewDesc{
            .format = Format::Rgba8Unorm,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        },
        colorTextureView);
    if (!result) {
        return result;
    }

    const uint64_t byteSize = static_cast<uint64_t>(newWidth) * static_cast<uint64_t>(newHeight) * 4ull;
    result = device->createBuffer(
        BufferDesc{
            .size = byteSize,
            .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback,
        },
        readbackBuffer);
    if (!result) {
        return result;
    }

    width = newWidth;
    height = newHeight;
    pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    return {};
}

Result TrianglePreviewRendererImpl::render(uint32_t newWidth, uint32_t newHeight)
{
    Result result = ensureResources(newWidth, newHeight);
    if (!result) {
        return result;
    }

    result = fence->wait();
    if (!result) {
        return result;
    }
    result = fence->reset();
    if (!result) {
        return result;
    }
    result = commandPool->reset();
    if (!result) {
        return result;
    }

    result = commandBuffer->begin();
    if (!result) {
        return result;
    }

    TextureBarrierDesc toColor{
        .texture = colorTexture.get(),
        .before = ResourceState::Undefined,
        .after = ResourceState::ColorAttachment,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer->barrier(BarrierDesc{.textures = &toColor, .textureCount = 1});

    const Rect renderArea{
        .x = 0,
        .y = 0,
        .width = width,
        .height = height,
    };
    RenderingAttachmentDesc colorAttachment{
        .view = colorTextureView.get(),
        .state = ResourceState::ColorAttachment,
        .loadOp = LoadOp::Clear,
        .storeOp = StoreOp::Store,
        .clearColor = ColorValue{0.04f, 0.06f, 0.09f, 1.0f},
    };
    commandBuffer->beginRendering(RenderingDesc{
        .renderArea = renderArea,
        .colorAttachments = &colorAttachment,
        .colorAttachmentCount = 1,
    });
    commandBuffer->setViewport(Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(width),
        .height = static_cast<float>(height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    });
    commandBuffer->setScissor(renderArea);
    commandBuffer->bindGraphicsPipeline(*pipeline);
    commandBuffer->draw(3);
    commandBuffer->endRendering();

    TextureBarrierDesc toTransfer{
        .texture = colorTexture.get(),
        .before = ResourceState::ColorAttachment,
        .after = ResourceState::TransferSource,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer->barrier(BarrierDesc{.textures = &toTransfer, .textureCount = 1});
    commandBuffer->copyTextureToBuffer(TextureBufferCopyDesc{
        .texture = colorTexture.get(),
        .buffer = readbackBuffer.get(),
        .width = width,
        .height = height,
        .depth = 1,
        .mipLevel = 0,
        .baseLayer = 0,
    });

    result = commandBuffer->end();
    if (!result) {
        return result;
    }

    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
    result = graphicsQueue->submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = fence.get(),
    });
    if (!result) {
        return result;
    }
    result = fence->wait();
    if (!result) {
        return result;
    }

    readbackBuffer->invalidate();
    void* mapped = readbackBuffer->map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }

    const uint64_t byteSize = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ull;
    std::memcpy(pixels.data(), mapped, static_cast<size_t>(byteSize));
    readbackBuffer->unmap();
    return {};
}

} // namespace detail

TrianglePreviewRenderer::TrianglePreviewRenderer()
    : impl_(std::make_unique<detail::TrianglePreviewRendererImpl>())
{
}

TrianglePreviewRenderer::~TrianglePreviewRenderer() = default;
TrianglePreviewRenderer::TrianglePreviewRenderer(TrianglePreviewRenderer&&) noexcept = default;
TrianglePreviewRenderer& TrianglePreviewRenderer::operator=(TrianglePreviewRenderer&&) noexcept = default;

Result TrianglePreviewRenderer::initialize(bool enableValidation)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return impl_->initialize(enableValidation);
}

Result TrianglePreviewRenderer::render(uint32_t width, uint32_t height)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return impl_->render(width, height);
}

const std::vector<uint32_t>& TrianglePreviewRenderer::pixels() const
{
    static const std::vector<uint32_t> emptyPixels;
    return impl_ != nullptr ? impl_->pixels : emptyPixels;
}

uint32_t TrianglePreviewRenderer::width() const
{
    return impl_ != nullptr ? impl_->width : 0;
}

uint32_t TrianglePreviewRenderer::height() const
{
    return impl_ != nullptr ? impl_->height : 0;
}

int runRhiTrianglePreviewTest(bool enableValidation)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        spdlog::error("SDL_Init failed: {}", SDL_GetError());
        return 1;
    }

    int exitCode = 0;
    {
        TrianglePreviewRenderer previewRenderer;
        Result result = previewRenderer.initialize(enableValidation);
        if (!checkResult(result, "TrianglePreviewRenderer::initialize")) {
            exitCode = resultToExitCode(result);
        } else {
            result = previewRenderer.render(320, 240);
            if (!checkResult(result, "TrianglePreviewRenderer::render")) {
                exitCode = resultToExitCode(result);
            } else {
                uint32_t brightPixelCount = 0;
                const std::vector<uint32_t>& pixels = previewRenderer.pixels();
                const auto* bytes = reinterpret_cast<const uint8_t*>(pixels.data());
                for (size_t index = 0; index < pixels.size(); ++index) {
                    const uint8_t r = bytes[index * 4 + 0];
                    const uint8_t g = bytes[index * 4 + 1];
                    const uint8_t b = bytes[index * 4 + 2];
                    if (r > 120 || g > 120 || b > 120) {
                        ++brightPixelCount;
                    }
                }

                if (brightPixelCount < 256) {
                    spdlog::error(
                        "Triangle preview pixel check failed: only {} bright pixels found.",
                        brightPixelCount);
                    exitCode = 1;
                }
            }
        }
    }

    SDL_Quit();
    return exitCode;
}

int runRhiBindlessDescriptorHeapSmokeTest(bool enableValidation)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        spdlog::error("SDL_Init failed: {}", SDL_GetError());
        return 1;
    }

    int exitCode = 0;
    {
        constexpr uint32_t kWidth = 16;
        constexpr uint32_t kHeight = 16;
        constexpr uint64_t kReadbackByteSize = static_cast<uint64_t>(kWidth) * kHeight * 4ull;

        std::unique_ptr<Device> device;
        std::unique_ptr<CommandPool> commandPool;
        std::unique_ptr<CommandBuffer> commandBuffer;
        std::unique_ptr<Fence> fence;
        std::unique_ptr<Texture> sourceTexture;
        std::unique_ptr<TextureView> sourceTextureView;
        std::unique_ptr<Texture> outputTexture;
        std::unique_ptr<TextureView> outputTextureView;
        std::unique_ptr<Buffer> readbackBuffer;
        std::unique_ptr<BindlessHeap> bindlessHeap;
        std::unique_ptr<ShaderModule> vertexShader;
        std::unique_ptr<ShaderModule> fragmentShader;
        std::unique_ptr<GraphicsPipeline> pipeline;

        Result result = createDevice(
            DeviceDesc{
                .applicationName = "Metallic RHI Bindless Descriptor Heap Smoke Test",
                .enableValidation = enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableAftermath = true,
            },
            device);
        if (!checkResult(result, "createDevice")) {
            exitCode = resultToExitCode(result);
        } else {
            const BindlessHeapDesc bindlessHeapDesc{
                .maxSampledImages = 1,
            };
            result = device->createBindlessHeap(bindlessHeapDesc, bindlessHeap);
            if (!device->capabilities().bindlessDescriptorHeap) {
                if (hasError(result, Error::Unsupported)) {
                    spdlog::info("VK_EXT_descriptor_heap unsupported; bindless smoke test skipped.");
                } else {
                    spdlog::error(
                        "createBindlessHeap was expected to return Unsupported, got {}",
                        resultToString(result));
                    exitCode = 1;
                }
            } else if (!checkResult(result, "createBindlessHeap")) {
                exitCode = resultToExitCode(result);
            } else {
                Queue* graphicsQueue = device->getQueue(QueueType::Graphics);
                if (graphicsQueue == nullptr) {
                    spdlog::error("No graphics queue available.");
                    exitCode = 1;
                }

                if (exitCode == 0) {
                    result = device->createCommandPool(*graphicsQueue, commandPool);
                    if (!checkResult(result, "createCommandPool")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = commandPool->createCommandBuffer(commandBuffer);
                    if (!checkResult(result, "createCommandBuffer")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createFence(true, fence);
                    if (!checkResult(result, "createFence")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createTexture(
                        TextureDesc{
                            .type = TextureType::Texture2D,
                            .usage = TextureUsageBits::Sampled | TextureUsageBits::ColorAttachment,
                            .format = Format::Rgba8Unorm,
                            .width = kWidth,
                            .height = kHeight,
                            .depth = 1,
                            .mipCount = 1,
                            .layerCount = 1,
                            .memoryLocation = MemoryLocation::Device,
                        },
                        sourceTexture);
                    if (!checkResult(result, "createTexture(source)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createTextureView(
                        *sourceTexture,
                        TextureViewDesc{
                            .format = Format::Rgba8Unorm,
                            .baseMip = 0,
                            .mipCount = 1,
                            .baseLayer = 0,
                            .layerCount = 1,
                        },
                        sourceTextureView);
                    if (!checkResult(result, "createTextureView(source)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createTexture(
                        TextureDesc{
                            .type = TextureType::Texture2D,
                            .usage = TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
                            .format = Format::Rgba8Unorm,
                            .width = kWidth,
                            .height = kHeight,
                            .depth = 1,
                            .mipCount = 1,
                            .layerCount = 1,
                            .memoryLocation = MemoryLocation::Device,
                        },
                        outputTexture);
                    if (!checkResult(result, "createTexture(output)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createTextureView(
                        *outputTexture,
                        TextureViewDesc{
                            .format = Format::Rgba8Unorm,
                            .baseMip = 0,
                            .mipCount = 1,
                            .baseLayer = 0,
                            .layerCount = 1,
                        },
                        outputTextureView);
                    if (!checkResult(result, "createTextureView(output)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createBuffer(
                        BufferDesc{
                            .size = kReadbackByteSize,
                            .usage = BufferUsageBits::TransferDestination,
                            .memoryLocation = MemoryLocation::HostReadback,
                        },
                        readbackBuffer);
                    if (!checkResult(result, "createBuffer(readback)")) {
                        exitCode = resultToExitCode(result);
                    }
                }

                BindlessHandle sourceImageHandle;
                if (exitCode == 0) {
                    result = bindlessHeap->allocateSampledImage(sourceImageHandle);
                    if (!checkResult(result, "allocateSampledImage")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = bindlessHeap->writeSampledImage(
                        sourceImageHandle,
                        *sourceTextureView,
                        ResourceState::ShaderRead);
                    if (!checkResult(result, "writeSampledImage")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = createSlangShaderModule(
                        *device,
                        kBindlessSmokeShaderModuleName,
                        kBindlessSmokeVertexEntryPoint,
                        vertexShader);
                    if (!checkResult(result, "createSlangShaderModule(vertex)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = createSlangShaderModule(
                        *device,
                        kBindlessSmokeShaderModuleName,
                        kBindlessSmokeFragmentEntryPoint,
                        fragmentShader);
                    if (!checkResult(result, "createSlangShaderModule(fragment)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createGraphicsPipeline(
                        GraphicsPipelineDesc{
                            .vertexShader = vertexShader.get(),
                            .fragmentShader = fragmentShader.get(),
                            .colorFormat = Format::Rgba8Unorm,
                            .topology = PrimitiveTopology::TriangleList,
                            .usesBindlessHeap = true,
                        },
                        pipeline);
                    if (!checkResult(result, "createGraphicsPipeline(bindless)")) {
                        exitCode = resultToExitCode(result);
                    }
                }

                if (exitCode == 0) {
                    result = fence->wait();
                    if (!checkResult(result, "fence wait")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = fence->reset();
                    if (!checkResult(result, "fence reset")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = commandPool->reset();
                    if (!checkResult(result, "commandPool reset")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = commandBuffer->begin();
                    if (!checkResult(result, "commandBuffer begin")) {
                        exitCode = resultToExitCode(result);
                    }
                }

                if (exitCode == 0) {
                    TextureBarrierDesc sourceToColor{
                        .texture = sourceTexture.get(),
                        .before = ResourceState::Undefined,
                        .after = ResourceState::ColorAttachment,
                        .baseMip = 0,
                        .mipCount = 1,
                        .baseLayer = 0,
                        .layerCount = 1,
                    };
                    commandBuffer->barrier(BarrierDesc{.textures = &sourceToColor, .textureCount = 1});

                    const Rect renderArea{
                        .x = 0,
                        .y = 0,
                        .width = kWidth,
                        .height = kHeight,
                    };
                    RenderingAttachmentDesc sourceAttachment{
                        .view = sourceTextureView.get(),
                        .state = ResourceState::ColorAttachment,
                        .loadOp = LoadOp::Clear,
                        .storeOp = StoreOp::Store,
                        .clearColor = ColorValue{0.25f, 0.50f, 0.75f, 1.0f},
                    };
                    commandBuffer->beginRendering(RenderingDesc{
                        .renderArea = renderArea,
                        .colorAttachments = &sourceAttachment,
                        .colorAttachmentCount = 1,
                    });
                    commandBuffer->endRendering();

                    TextureBarrierDesc sourceToShaderRead{
                        .texture = sourceTexture.get(),
                        .before = ResourceState::ColorAttachment,
                        .after = ResourceState::ShaderRead,
                        .baseMip = 0,
                        .mipCount = 1,
                        .baseLayer = 0,
                        .layerCount = 1,
                    };
                    commandBuffer->barrier(BarrierDesc{.textures = &sourceToShaderRead, .textureCount = 1});

                    TextureBarrierDesc outputToColor{
                        .texture = outputTexture.get(),
                        .before = ResourceState::Undefined,
                        .after = ResourceState::ColorAttachment,
                        .baseMip = 0,
                        .mipCount = 1,
                        .baseLayer = 0,
                        .layerCount = 1,
                    };
                    commandBuffer->barrier(BarrierDesc{.textures = &outputToColor, .textureCount = 1});

                    RenderingAttachmentDesc outputAttachment{
                        .view = outputTextureView.get(),
                        .state = ResourceState::ColorAttachment,
                        .loadOp = LoadOp::Clear,
                        .storeOp = StoreOp::Store,
                        .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
                    };
                    commandBuffer->beginRendering(RenderingDesc{
                        .renderArea = renderArea,
                        .colorAttachments = &outputAttachment,
                        .colorAttachmentCount = 1,
                    });
                    commandBuffer->setViewport(Viewport{
                        .x = 0.0f,
                        .y = 0.0f,
                        .width = static_cast<float>(kWidth),
                        .height = static_cast<float>(kHeight),
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f,
                    });
                    commandBuffer->setScissor(renderArea);
                    commandBuffer->bindGraphicsPipeline(*pipeline);
                    commandBuffer->bindBindlessHeap(*bindlessHeap);
                    commandBuffer->draw(3);
                    commandBuffer->endRendering();

                    TextureBarrierDesc outputToTransfer{
                        .texture = outputTexture.get(),
                        .before = ResourceState::ColorAttachment,
                        .after = ResourceState::TransferSource,
                        .baseMip = 0,
                        .mipCount = 1,
                        .baseLayer = 0,
                        .layerCount = 1,
                    };
                    commandBuffer->barrier(BarrierDesc{.textures = &outputToTransfer, .textureCount = 1});
                    commandBuffer->copyTextureToBuffer(TextureBufferCopyDesc{
                        .texture = outputTexture.get(),
                        .buffer = readbackBuffer.get(),
                        .width = kWidth,
                        .height = kHeight,
                        .depth = 1,
                        .mipLevel = 0,
                        .baseLayer = 0,
                    });

                    result = commandBuffer->end();
                    if (!checkResult(result, "commandBuffer end")) {
                        exitCode = resultToExitCode(result);
                    }
                }

                if (exitCode == 0) {
                    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
                    result = graphicsQueue->submit(QueueSubmitDesc{
                        .commandBuffers = commandBuffers,
                        .commandBufferCount = 1,
                        .signalFence = fence.get(),
                    });
                    if (!checkResult(result, "graphicsQueue submit")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = fence->wait();
                    if (!checkResult(result, "fence wait after submit")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    readbackBuffer->invalidate();
                    std::vector<uint32_t> pixels(static_cast<size_t>(kWidth) * kHeight);
                    void* mapped = readbackBuffer->map();
                    if (mapped == nullptr) {
                        spdlog::error("Failed to map bindless smoke readback buffer.");
                        exitCode = 1;
                    } else {
                        std::memcpy(pixels.data(), mapped, static_cast<size_t>(kReadbackByteSize));
                        readbackBuffer->unmap();

                        uint32_t matchedPixelCount = 0;
                        const auto* bytes = reinterpret_cast<const uint8_t*>(pixels.data());
                        for (size_t index = 0; index < pixels.size(); ++index) {
                            const uint8_t r = bytes[index * 4 + 0];
                            const uint8_t g = bytes[index * 4 + 1];
                            const uint8_t b = bytes[index * 4 + 2];
                            const uint8_t a = bytes[index * 4 + 3];
                            if (r >= 48 && r <= 80 && g >= 112 && g <= 144 && b >= 176 && b <= 208 && a >= 240) {
                                ++matchedPixelCount;
                            }
                        }

                        if (matchedPixelCount < pixels.size() / 2) {
                            const uint8_t r = bytes[0];
                            const uint8_t g = bytes[1];
                            const uint8_t b = bytes[2];
                            const uint8_t a = bytes[3];
                            spdlog::error(
                                "Bindless descriptor heap pixel check failed: {} matching pixels. First pixel RGBA=({}, {}, {}, {}).",
                                matchedPixelCount,
                                static_cast<uint32_t>(r),
                                static_cast<uint32_t>(g),
                                static_cast<uint32_t>(b),
                                static_cast<uint32_t>(a));
                            exitCode = 1;
                        }
                    }
                }
            }

            if (device != nullptr) {
                (void)device->waitIdle();
            }
        }
    }

    SDL_Quit();
    return exitCode;
}

int runRhiSmokeTest(bool enableValidation)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        spdlog::error("SDL_Init failed: {}", SDL_GetError());
        return 1;
    }

    const SDL_WindowFlags windowFlags =
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow("Metallic RHI Smoke Test", 1280, 720, windowFlags);
    if (window == nullptr) {
        spdlog::error("SDL_CreateWindow failed: {}", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    std::unique_ptr<Device> device;
    std::unique_ptr<Swapchain> swapchain;
    std::vector<std::unique_ptr<TextureView>> swapchainViews;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<SwapchainSemaphore> imageAvailable;
    std::vector<std::unique_ptr<SwapchainSemaphore>> renderFinishedSemaphores;
    std::unique_ptr<Fence> frameFence;

    auto cleanup = [&]() {
        if (device != nullptr) {
            (void)device->waitIdle();
        }

        swapchainViews.clear();
        renderFinishedSemaphores.clear();
        commandBuffer.reset();
        commandPool.reset();
        imageAvailable.reset();
        frameFence.reset();
        swapchain.reset();
        device.reset();

        if (window != nullptr) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        SDL_Quit();
    };

    int pixelWidth = 0;
    int pixelHeight = 0;
    if (!SDL_GetWindowSizeInPixels(window, &pixelWidth, &pixelHeight)) {
        spdlog::error("SDL_GetWindowSizeInPixels failed: {}", SDL_GetError());
        cleanup();
        return 1;
    }

    Result result = createDevice(
        DeviceDesc{
            .applicationName = "Metallic RHI Smoke Test",
            .enableValidation = enableValidation,
            .enableAftermath = true,
        },
        device);
    if (!checkResult(result, "createDevice")) {
        cleanup();
        return resultToExitCode(result);
    }

    Queue* graphicsQueue = device->getQueue(QueueType::Graphics);
    if (graphicsQueue == nullptr) {
        spdlog::error("No graphics queue available.");
        cleanup();
        return 1;
    }

    result = device->createSwapchain(
        SwapchainDesc{
            .window = {
                .system = WindowSystem::Sdl3,
                .nativeWindow = window,
            },
            .width = static_cast<uint32_t>(std::max(pixelWidth, 1)),
            .height = static_cast<uint32_t>(std::max(pixelHeight, 1)),
            .imageCount = 3,
            .framesInFlight = 2,
            .format = Format::Bgra8Srgb,
            .vsync = true,
        },
        swapchain);
    if (!checkResult(result, "createSwapchain")) {
        cleanup();
        return resultToExitCode(result);
    }

    swapchainViews.reserve(swapchain->imageCount());
    renderFinishedSemaphores.reserve(swapchain->imageCount());
    for (uint32_t imageIndex = 0; imageIndex < swapchain->imageCount(); ++imageIndex) {
        std::unique_ptr<TextureView> view;
        result = device->createTextureView(
            *swapchain->texture(imageIndex),
            TextureViewDesc{
                .format = swapchain->format(),
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            view);
        if (!checkResult(result, "createTextureView")) {
            cleanup();
            return resultToExitCode(result);
        }
        swapchainViews.push_back(std::move(view));

        std::unique_ptr<SwapchainSemaphore> renderFinished;
        result = device->createSwapchainSemaphore(renderFinished);
        if (!checkResult(result, "createSwapchainSemaphore(renderFinished)")) {
            cleanup();
            return resultToExitCode(result);
        }
        renderFinishedSemaphores.push_back(std::move(renderFinished));
    }

    result = device->createCommandPool(*graphicsQueue, commandPool);
    if (!checkResult(result, "createCommandPool")) {
        cleanup();
        return resultToExitCode(result);
    }

    result = commandPool->createCommandBuffer(commandBuffer);
    if (!checkResult(result, "createCommandBuffer")) {
        cleanup();
        return resultToExitCode(result);
    }

    if (!checkResult(device->createSwapchainSemaphore(imageAvailable), "createSwapchainSemaphore(imageAvailable)") ||
        !checkResult(device->createFence(false, frameFence), "createFence")) {
        cleanup();
        return 1;
    }

    uint32_t imageIndex = 0;
    result = swapchain->acquireNextImage(*imageAvailable, imageIndex);
    if (!checkResult(result, "acquireNextImage")) {
        cleanup();
        return resultToExitCode(result);
    }
    if (imageIndex >= renderFinishedSemaphores.size() || renderFinishedSemaphores[imageIndex] == nullptr) {
        spdlog::error("acquireNextImage returned invalid image index.");
        cleanup();
        return 1;
    }

    result = commandBuffer->begin();
    if (!checkResult(result, "CommandBuffer::begin")) {
        cleanup();
        return resultToExitCode(result);
    }

    TextureBarrierDesc toColor{
        .texture = swapchain->texture(imageIndex),
        .before = ResourceState::Undefined,
        .after = ResourceState::ColorAttachment,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer->barrier(BarrierDesc{.textures = &toColor, .textureCount = 1});

    const Rect renderArea{
        .x = 0,
        .y = 0,
        .width = swapchain->width(),
        .height = swapchain->height(),
    };
    const ColorValue clearColor{0.04f, 0.08f, 0.13f, 1.0f};
    RenderingAttachmentDesc colorAttachment{
        .view = swapchainViews[imageIndex].get(),
        .state = ResourceState::ColorAttachment,
        .loadOp = LoadOp::DontCare,
        .storeOp = StoreOp::Store,
        .clearColor = clearColor,
    };
    commandBuffer->beginRendering(RenderingDesc{
        .renderArea = renderArea,
        .colorAttachments = &colorAttachment,
        .colorAttachmentCount = 1,
    });
    commandBuffer->clearColorAttachment(0, clearColor, renderArea);
    commandBuffer->endRendering();

    TextureBarrierDesc toPresent{
        .texture = swapchain->texture(imageIndex),
        .before = ResourceState::ColorAttachment,
        .after = ResourceState::Present,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer->barrier(BarrierDesc{.textures = &toPresent, .textureCount = 1});

    result = commandBuffer->end();
    if (!checkResult(result, "CommandBuffer::end")) {
        cleanup();
        return resultToExitCode(result);
    }

    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
    SwapchainSemaphoreSubmitDesc waitSemaphore{
        .semaphore = imageAvailable.get(),
        .stages = PipelineStageBits::ColorAttachment,
    };
    SwapchainSemaphoreSubmitDesc signalSemaphore{
        .semaphore = renderFinishedSemaphores[imageIndex].get(),
        .stages = PipelineStageBits::AllCommands,
    };
    result = graphicsQueue->submit(QueueSubmitDesc{
        .waitSwapchainSemaphores = &waitSemaphore,
        .waitSwapchainSemaphoreCount = 1,
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalSwapchainSemaphores = &signalSemaphore,
        .signalSwapchainSemaphoreCount = 1,
        .signalFence = frameFence.get(),
    });
    if (!checkResult(result, "Queue::submit")) {
        cleanup();
        return resultToExitCode(result);
    }

    result = swapchain->present(*graphicsQueue, imageIndex, *renderFinishedSemaphores[imageIndex]);
    if (!checkResult(result, "Swapchain::present")) {
        cleanup();
        return resultToExitCode(result);
    }

    result = frameFence->wait();
    if (!checkResult(result, "Fence::wait")) {
        cleanup();
        return resultToExitCode(result);
    }

    cleanup();
    return 0;
}

} // namespace metallic::render
