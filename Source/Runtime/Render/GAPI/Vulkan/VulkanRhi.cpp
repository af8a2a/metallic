#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/SlangCompiler.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_VULKAN_VERSION 1004000
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

constexpr uint32_t kVulkanApiVersion = VK_API_VERSION_1_4;
constexpr uint64_t kAcquireTimeoutNanoseconds = std::numeric_limits<uint64_t>::max();

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

bool acquireSdlVulkanLibrary()
{
    std::lock_guard lock(sdlVulkanLibraryMutex());
    uint32_t& refCount = sdlVulkanLibraryRefCount();
    if (refCount == 0 && !SDL_Vulkan_LoadLibrary(nullptr)) {
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

bool hasDeviceExtension(VkPhysicalDevice physicalDevice, const char* extensionName)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    if (count > 0) {
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data());
    }
    return hasName(extensions, extensionName);
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
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
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
        std::cerr << "Vulkan validation: " << callbackData->pMessage << '\n';
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
            if (handle.index < maxImages_) {
                freeImageSlots_.push_back(handle.index);
            }
            break;
        case BindlessHandleKind::Buffer:
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

    VkResult writeImageDescriptor(
        BindlessHandle handle,
        VkImage image,
        VkFormat format,
        VkImageLayout layout,
        const VkImageSubresourceRange& subresourceRange,
        VkImageViewType viewType,
        void* resourceHeapBase)
    {
        if (handle.kind != BindlessHandleKind::SampledImage ||
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
    QueueType type = QueueType::Graphics;
};

struct FenceImpl {
    DeviceImpl* device = nullptr;
    VkFence fence = VK_NULL_HANDLE;
};

struct SemaphoreImpl {
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
    VmaAllocation allocation = VK_NULL_HANDLE;
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
};

struct GraphicsPipelineImpl {
    DeviceImpl* device = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool usesBindlessHeap = false;
};

struct ComputePipelineImpl {
    DeviceImpl* device = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool usesBindlessHeap = false;
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
    VkPipelineLayout currentGraphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout currentComputePipelineLayout = VK_NULL_HANDLE;
    BindlessHeapImpl* currentBindlessHeap = nullptr;
    bool currentGraphicsPipelineUsesBindlessHeap = false;
    bool currentComputePipelineUsesBindlessHeap = false;
    bool currentGraphicsShaderObjectUsesBindlessHeap = false;
    bool currentGraphicsShaderObjectBound = false;
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
    DescriptorHeapWriter descriptorHeapWriter;
    uint32_t graphicsFamily = 0;
    uint32_t computeFamily = 0;
    bool sdlVulkanLoaded = false;
    bool validationEnabled = false;
    bool debugUtilsEnabled = false;
    bool bindlessDescriptorHeapEnabled = false;
    bool shaderObjectEnabled = false;
    bool bufferDeviceAddressEnabled = false;
    bool rayTracingAccelerationStructureEnabled = false;
    bool rayQueryEnabled = false;
    bool pushDescriptorEnabled = false;
    PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabel = nullptr;
    std::vector<std::unique_ptr<Queue>> queues;

    ~DeviceImpl();
    void addQueue(VkQueue queue, uint32_t familyIndex, QueueType type);
};

DeviceImpl::~DeviceImpl()
{
    queues.clear();

    if (device != VK_NULL_HANDLE) {
        activateVolkDevice(device);
        vkDeviceWaitIdle(device);
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
}

void DeviceImpl::addQueue(VkQueue queue, uint32_t familyIndex, QueueType type)
{
    auto impl = std::make_unique<QueueImpl>();
    impl->device = this;
    impl->queue = queue;
    impl->familyIndex = familyIndex;
    impl->type = type;
    queues.emplace_back(new Queue(std::move(impl)));
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
    if (heapDesc.maxSampledImages == 0 && heapDesc.maxBuffers == 0) {
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

    if (heap.setupResourceHeap(desc.maxSampledImages, desc.maxBuffers) == 0) {
        return makeError(Error::Unsupported);
    }
    return createHeapBuffer(heap.resourceHeapSize(), heap.resourceHeapAlignment(), resourceHeap);
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
        std::cerr << "SDL_Vulkan_CreateSurface failed: " << SDL_GetError() << '\n';
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
        (desc.commandBufferCount > 0 && desc.commandBuffers == nullptr) ||
        (desc.signalSemaphoreCount > 0 && desc.signalSemaphores == nullptr)) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<VkSemaphoreSubmitInfo> waitSemaphores;
    waitSemaphores.reserve(desc.waitSemaphoreCount);
    for (uint32_t index = 0; index < desc.waitSemaphoreCount; ++index) {
        const SemaphoreSubmitDesc& wait = desc.waitSemaphores[index];
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
    signalSemaphores.reserve(desc.signalSemaphoreCount);
    for (uint32_t index = 0; index < desc.signalSemaphoreCount; ++index) {
        const SemaphoreSubmitDesc& signal = desc.signalSemaphores[index];
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

Result BindlessHeap::writeSampledImage(BindlessHandle handle, TextureView& view, ResourceState state)
{
    if (impl_ == nullptr ||
        impl_->resourceHeap.mapped == nullptr ||
        view.impl_ == nullptr ||
        view.impl_->texture == nullptr ||
        view.impl_->texture->impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (handle.kind != BindlessHandleKind::SampledImage) {
        return makeError(Error::InvalidArgument);
    }

    const TextureDesc& textureDesc = view.impl_->texture->impl_->desc;
    const TextureViewDesc& viewDesc = view.impl_->desc;
    const VkImageSubresourceRange subresourceRange{
        .aspectMask = aspectForFormat(textureDesc.format),
        .baseMipLevel = viewDesc.baseMip,
        .levelCount = viewDesc.mipCount,
        .baseArrayLayer = viewDesc.baseLayer,
        .layerCount = viewDesc.layerCount,
    };
    const StateInfo imageState = stateInfo(state);
    const VkResult result = impl_->heap.writeImageDescriptor(
        handle,
        view.impl_->texture->impl_->image,
        view.impl_->format,
        imageState.layout,
        subresourceRange,
        toVkImageViewType(textureDesc.type),
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
    impl_->currentBindlessHeap = nullptr;
    impl_->currentGraphicsPipelineUsesBindlessHeap = false;
    impl_->currentComputePipelineUsesBindlessHeap = false;
    impl_->currentGraphicsShaderObjectUsesBindlessHeap = false;
    impl_->currentGraphicsShaderObjectBound = false;
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
        desc.depth == 0) {
        return;
    }

    VkBufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = aspectForFormat(desc.texture->impl_->desc.format),
            .mipLevel = desc.mipLevel,
            .baseArrayLayer = desc.baseLayer,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
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
        desc.depth == 0) {
        return;
    }

    VkBufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = aspectForFormat(desc.texture->impl_->desc.format),
            .mipLevel = desc.mipLevel,
            .baseArrayLayer = desc.baseLayer,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
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
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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

    const size_t payloadSize = sizeof(push) + commandBuffer.currentBindlessUserData.size();
    const bool needsDescriptorHeapPush =
        commandBuffer.currentGraphicsPipelineUsesBindlessHeap ||
        commandBuffer.currentComputePipelineUsesBindlessHeap ||
        commandBuffer.currentGraphicsShaderObjectUsesBindlessHeap;
    if (needsDescriptorHeapPush &&
        commandBuffer.device != nullptr &&
        commandBuffer.device->bindlessDescriptorHeapEnabled &&
        commandBuffer.device->descriptorHeapWriter.maxPushDataSize() >= payloadSize &&
        vkCmdPushDataEXT != nullptr) {
        std::vector<uint8_t> payload(payloadSize);
        std::memcpy(payload.data(), &push, sizeof(push));
        if (!commandBuffer.currentBindlessUserData.empty()) {
            std::memcpy(
                payload.data() + sizeof(push),
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
    impl_->currentGraphicsPipelineUsesBindlessHeap = pipeline.impl_->usesBindlessHeap;
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
    if (impl_->currentGraphicsShaderObjectUsesBindlessHeap && impl_->currentBindlessHeap != nullptr) {
        pushCurrentBindlessData(*impl_, *impl_->currentBindlessHeap);
    }
}

void CommandBuffer::bindBindlessHeap(BindlessHeap& heap)
{
    if (impl_ == nullptr || heap.impl_ == nullptr) {
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

void CommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    if (impl_ != nullptr && groupCountX > 0 && groupCountY > 0 && groupCountZ > 0) {
        vkCmdDispatch(impl_->commandBuffer, groupCountX, groupCountY, groupCountZ);
    }
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

Result Swapchain::acquireNextImage(Semaphore& semaphore, uint32_t& imageIndex)
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

Result Swapchain::present(Queue& queue, uint32_t imageIndex, Semaphore& waitSemaphore)
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

Result Device::createSemaphore(std::unique_ptr<Semaphore>& outSemaphore)
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

    auto semaphoreImpl = std::make_unique<detail::SemaphoreImpl>();
    semaphoreImpl->device = impl_.get();
    semaphoreImpl->semaphore = semaphore;
    outSemaphore.reset(new Semaphore(std::move(semaphoreImpl)));
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

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = desc.size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
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
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo allocationInfo = allocationInfoForMemory(desc.memoryLocation);
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkResult result = vmaCreateImage(
        impl_->allocator,
        &imageInfo,
        &allocationInfo,
        &image,
        &allocation,
        nullptr);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }

    auto textureImpl = std::make_unique<detail::TextureImpl>();
    textureImpl->device = impl_.get();
    textureImpl->desc = desc;
    textureImpl->image = image;
    textureImpl->allocation = allocation;
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

    auto shaderImpl = std::make_unique<detail::ShaderModuleImpl>();
    shaderImpl->device = impl_.get();
    shaderImpl->module = module;
    outShaderModule.reset(new ShaderModule(std::move(shaderImpl)));
    return {};
}

Result Device::createGraphicsPipeline(
    const GraphicsPipelineDesc& desc,
    std::unique_ptr<GraphicsPipeline>& outGraphicsPipeline)
{
    outGraphicsPipeline.reset();
    if (impl_ == nullptr ||
        desc.vertexShader == nullptr ||
        desc.vertexShader->impl_ == nullptr ||
        desc.fragmentShader == nullptr ||
        desc.fragmentShader->impl_ == nullptr ||
        desc.colorFormat == Format::Unknown) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);
    if (desc.usesBindlessHeap && !impl_->capabilities.bindlessDescriptorHeap) {
        return makeError(Error::Unsupported);
    }
    const bool hasDepthStencilFormat = desc.depthStencilFormat != Format::Unknown;
    if ((desc.depthStencil.depthTestEnable || desc.depthStencil.depthWriteEnable) && !hasDepthStencilFormat) {
        return makeError(Error::InvalidArgument);
    }

    const char* vertexEntryPoint = desc.vertexEntryPoint != nullptr ? desc.vertexEntryPoint : "main";
    const char* fragmentEntryPoint = desc.fragmentEntryPoint != nullptr ? desc.fragmentEntryPoint : "main";
    std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = desc.vertexShader->impl_->module,
            .pName = vertexEntryPoint,
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = desc.fragmentShader->impl_->module,
            .pName = fragmentEntryPoint,
        },
    };
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
        stages[0].pNext = &bindlessMappingInfo;
        stages[1].pNext = &bindlessMappingInfo;
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
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
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
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizationState,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = hasDepthStencilFormat ? &depthStencilState : nullptr,
        .pColorBlendState = &colorBlendState,
        .pDynamicState = &dynamicState,
        .layout = layout,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vkCreateGraphicsPipelines(impl_->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
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
    pipelineImpl->usesBindlessHeap = desc.usesBindlessHeap;
    outGraphicsPipeline.reset(new GraphicsPipeline(std::move(pipelineImpl)));
    return {};
}

Result Device::createComputePipeline(
    const ComputePipelineDesc& desc,
    std::unique_ptr<ComputePipeline>& outComputePipeline)
{
    outComputePipeline.reset();
    if (impl_ == nullptr || desc.computeShader == nullptr || desc.computeShader->impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    activateVolkDevice(impl_->device);
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

    const char* computeEntryPoint = desc.computeEntryPoint != nullptr ? desc.computeEntryPoint : "main";
    VkPipelineShaderStageCreateInfo stage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = desc.computeShader->impl_->module,
        .pName = computeEntryPoint,
    };

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
    result = vkCreateComputePipelines(impl_->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
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
    if (!acquireSdlVulkanLibrary()) {
        std::cerr << "SDL_Vulkan_LoadLibrary failed: " << SDL_GetError() << '\n';
        return makeError(Error::Unsupported);
    }
    deviceImpl->sdlVulkanLoaded = true;

    VkResult vkResult = volkInitialize();
    if (vkResult != VK_SUCCESS) {
        std::cerr << "volkInitialize failed with VkResult " << static_cast<int>(vkResult) << '\n';
        return resultFromVk(vkResult);
    }

    Uint32 sdlExtensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    if (sdlExtensions == nullptr || sdlExtensionCount == 0) {
        std::cerr << "SDL_Vulkan_GetInstanceExtensions failed: " << SDL_GetError() << '\n';
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

    std::vector<const char*> instanceLayers;
    const std::vector<VkLayerProperties> availableLayers = enumerateInstanceLayers();
    if (desc.enableValidation && hasName(availableLayers, "VK_LAYER_KHRONOS_validation")) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
        deviceImpl->validationEnabled = true;
    } else if (desc.enableValidation) {
        std::cerr << "Vulkan validation requested but VK_LAYER_KHRONOS_validation is not available.\n";
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

    const bool requestBindlessDescriptorHeap = desc.enableBindlessDescriptorHeap;
    const bool requestShaderObject = desc.enableShaderObject;
    const bool requestRayTracingAccelerationStructure = desc.enableRayTracingAccelerationStructure;
    const bool requestRayQuery = desc.enableRayQuery;
    const bool requestPushDescriptor = desc.enablePushDescriptor;
    VkPhysicalDevice bestPhysicalDevice = VK_NULL_HANDLE;
    uint32_t bestGraphicsFamily = 0;
    uint32_t bestComputeFamily = 0;
    int32_t bestFeatureScore = -1;
    bool bestBindlessDescriptorHeap = false;
    bool bestShaderObject = false;
    bool bestRayTracingAccelerationStructure = false;
    bool bestRayQuery = false;
    bool bestPushDescriptor = false;
    bool selectedBindlessDescriptorHeap = false;
    bool selectedShaderObject = false;
    bool selectedRayTracingAccelerationStructure = false;
    bool selectedRayQuery = false;
    bool selectedPushDescriptor = false;

    for (VkPhysicalDevice physicalDevice : physicalDevices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        if (properties.apiVersion < kVulkanApiVersion) {
            continue;
        }

        const bool swapchainExtensionAvailable = hasDeviceExtension(physicalDevice, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        const bool descriptorHeapExtensionAvailable =
            hasDeviceExtension(physicalDevice, VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
        const bool shaderObjectExtensionAvailable =
            hasDeviceExtension(physicalDevice, VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
        const bool accelerationStructureExtensionAvailable =
            hasDeviceExtension(physicalDevice, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        const bool deferredHostOperationsExtensionAvailable =
            hasDeviceExtension(physicalDevice, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        const bool rayQueryExtensionAvailable =
            hasDeviceExtension(physicalDevice, VK_KHR_RAY_QUERY_EXTENSION_NAME);
        const bool pushDescriptorExtensionAvailable =
            hasDeviceExtension(physicalDevice, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        if (!swapchainExtensionAvailable) {
            continue;
        }

        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        };
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
            .pNext = &accelerationStructureFeatures,
        };
        VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
            .pNext = &rayQueryFeatures,
        };
        VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptorHeapFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
            .pNext = &shaderObjectFeatures,
        };
        VkPhysicalDeviceVulkan13Features vulkan13Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &descriptorHeapFeatures,
        };
        VkPhysicalDeviceVulkan12Features vulkan12Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &vulkan13Features,
        };
        VkPhysicalDeviceVulkan11Features vulkan11Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext = &vulkan12Features,
        };
        VkPhysicalDeviceFeatures2 features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vulkan11Features,
        };
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
        if (vulkan11Features.shaderDrawParameters != VK_TRUE ||
            vulkan13Features.dynamicRendering != VK_TRUE ||
            vulkan13Features.synchronization2 != VK_TRUE) {
            continue;
        }

        const bool descriptorHeapSupported =
            requestBindlessDescriptorHeap &&
            descriptorHeapExtensionAvailable &&
            descriptorHeapFeatures.descriptorHeap == VK_TRUE &&
            vulkan12Features.descriptorIndexing == VK_TRUE &&
            vulkan12Features.runtimeDescriptorArray == VK_TRUE &&
            vulkan12Features.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
            vulkan12Features.bufferDeviceAddress == VK_TRUE &&
            DescriptorHeapWriter::isSupported(physicalDevice);
        const bool shaderObjectSupported =
            requestShaderObject &&
            shaderObjectExtensionAvailable &&
            shaderObjectFeatures.shaderObject == VK_TRUE;
        const bool accelerationStructureSupported =
            accelerationStructureExtensionAvailable &&
            deferredHostOperationsExtensionAvailable &&
            accelerationStructureFeatures.accelerationStructure == VK_TRUE &&
            vulkan12Features.bufferDeviceAddress == VK_TRUE;
        const bool rayTracingAccelerationStructureSupported =
            (requestRayTracingAccelerationStructure || requestRayQuery) &&
            accelerationStructureSupported;
        const bool rayQuerySupported =
            requestRayQuery &&
            accelerationStructureSupported &&
            rayQueryExtensionAvailable &&
            rayQueryFeatures.rayQuery == VK_TRUE;
        const bool pushDescriptorSupported =
            requestPushDescriptor &&
            pushDescriptorExtensionAvailable;

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

        const bool matchesRequestedFeatures =
            (!requestBindlessDescriptorHeap || descriptorHeapSupported) &&
            (!requestShaderObject || shaderObjectSupported) &&
            (!requestRayTracingAccelerationStructure || rayTracingAccelerationStructureSupported) &&
            (!requestRayQuery || rayQuerySupported) &&
            (!requestPushDescriptor || pushDescriptorSupported);
        const int32_t featureScore =
            (descriptorHeapSupported ? 16 : 0) +
            (shaderObjectSupported ? 8 : 0) +
            (rayTracingAccelerationStructureSupported ? 4 : 0) +
            (rayQuerySupported ? 2 : 0) +
            (pushDescriptorSupported ? 1 : 0);
        if (featureScore > bestFeatureScore) {
            bestFeatureScore = featureScore;
            bestPhysicalDevice = physicalDevice;
            bestGraphicsFamily = graphicsFamily;
            bestComputeFamily = computeFamily;
            bestBindlessDescriptorHeap = descriptorHeapSupported;
            bestShaderObject = shaderObjectSupported;
            bestRayTracingAccelerationStructure = rayTracingAccelerationStructureSupported;
            bestRayQuery = rayQuerySupported;
            bestPushDescriptor = pushDescriptorSupported;
        }
        if (matchesRequestedFeatures) {
            deviceImpl->physicalDevice = physicalDevice;
            deviceImpl->graphicsFamily = graphicsFamily;
            deviceImpl->computeFamily = computeFamily;
            selectedBindlessDescriptorHeap = descriptorHeapSupported;
            selectedShaderObject = shaderObjectSupported;
            selectedRayTracingAccelerationStructure =
                (requestRayTracingAccelerationStructure || requestRayQuery) &&
                accelerationStructureSupported;
            selectedRayQuery = rayQuerySupported;
            selectedPushDescriptor = pushDescriptorSupported;
            break;
        }
    }

    if (deviceImpl->physicalDevice == VK_NULL_HANDLE && bestPhysicalDevice != VK_NULL_HANDLE) {
        deviceImpl->physicalDevice = bestPhysicalDevice;
        deviceImpl->graphicsFamily = bestGraphicsFamily;
        deviceImpl->computeFamily = bestComputeFamily;
        selectedBindlessDescriptorHeap = bestBindlessDescriptorHeap;
        selectedShaderObject = bestShaderObject;
        selectedRayTracingAccelerationStructure = bestRayTracingAccelerationStructure;
        selectedRayQuery = bestRayQuery;
        selectedPushDescriptor = bestPushDescriptor;
    }

    if (deviceImpl->physicalDevice == VK_NULL_HANDLE) {
        return makeError(Error::Unsupported);
    }

    const float queuePriority = 1.0f;
    std::array<uint32_t, 2> queueFamilies = {
        deviceImpl->graphicsFamily,
        deviceImpl->computeFamily,
    };
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(queueFamilies.size());
    for (uint32_t queueFamily : queueFamilies) {
        const bool alreadyAdded = std::any_of(
            queueInfos.begin(),
            queueInfos.end(),
            [queueFamily](const VkDeviceQueueCreateInfo& info) {
                return info.queueFamilyIndex == queueFamily;
            });
        if (alreadyAdded) {
            continue;
        }
        queueInfos.push_back({
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        });
    }

    VkPhysicalDeviceVulkan13Features enabledVulkan13Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceDescriptorHeapFeaturesEXT enabledDescriptorHeapFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
        .descriptorHeap = selectedBindlessDescriptorHeap ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceShaderObjectFeaturesEXT enabledShaderObjectFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
        .shaderObject = selectedShaderObject ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .accelerationStructure = selectedRayTracingAccelerationStructure ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceRayQueryFeaturesKHR enabledRayQueryFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .rayQuery = selectedRayQuery ? VK_TRUE : VK_FALSE,
    };
    void** featureTail = &enabledVulkan13Features.pNext;
    if (selectedBindlessDescriptorHeap) {
        *featureTail = &enabledDescriptorHeapFeatures;
        featureTail = &enabledDescriptorHeapFeatures.pNext;
    }
    if (selectedShaderObject) {
        *featureTail = &enabledShaderObjectFeatures;
        featureTail = &enabledShaderObjectFeatures.pNext;
    }
    if (selectedRayTracingAccelerationStructure) {
        *featureTail = &enabledAccelerationStructureFeatures;
        featureTail = &enabledAccelerationStructureFeatures.pNext;
    }
    if (selectedRayQuery) {
        *featureTail = &enabledRayQueryFeatures;
    }
    VkPhysicalDeviceVulkan12Features enabledVulkan12Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &enabledVulkan13Features,
        .descriptorIndexing = selectedBindlessDescriptorHeap ? VK_TRUE : VK_FALSE,
        .shaderSampledImageArrayNonUniformIndexing = selectedBindlessDescriptorHeap ? VK_TRUE : VK_FALSE,
        .runtimeDescriptorArray = selectedBindlessDescriptorHeap ? VK_TRUE : VK_FALSE,
        .bufferDeviceAddress =
            (selectedBindlessDescriptorHeap || selectedRayTracingAccelerationStructure || selectedRayQuery)
                ? VK_TRUE
                : VK_FALSE,
    };
    VkPhysicalDeviceVulkan11Features enabledVulkan11Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &enabledVulkan12Features,
        .shaderDrawParameters = VK_TRUE,
    };
    VkPhysicalDeviceFeatures2 enabledFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &enabledVulkan11Features,
    };

    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    if (selectedBindlessDescriptorHeap) {
        deviceExtensions.push_back(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
    }
    if (selectedShaderObject) {
        deviceExtensions.push_back(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
    }
    if (selectedRayTracingAccelerationStructure) {
        deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    }
    if (selectedRayQuery) {
        deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }
    if (selectedPushDescriptor) {
        deviceExtensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    }

    VkDeviceCreateInfo deviceInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &enabledFeatures,
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

    if (selectedBindlessDescriptorHeap) {
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

        deviceImpl->capabilities = DeviceCapabilities{
            .bindlessDescriptorHeap = true,
            .maxBindlessSamplers = capacityFromBytes(
                samplerCapacityBytes,
                deviceImpl->descriptorHeapWriter.samplerDescriptorSize()),
            .maxBindlessSampledImages = capacityFromBytes(
                resourceCapacityBytes,
                deviceImpl->descriptorHeapWriter.imageDescriptorSize()),
            .maxBindlessBuffers = capacityFromBytes(
                resourceCapacityBytes,
                deviceImpl->descriptorHeapWriter.bufferDescriptorSize()),
        };
        deviceImpl->bindlessDescriptorHeapEnabled = true;
    }
    deviceImpl->capabilities.shaderObject = selectedShaderObject;
    deviceImpl->shaderObjectEnabled = selectedShaderObject;
    deviceImpl->capabilities.rayTracingAccelerationStructure = selectedRayTracingAccelerationStructure;
    deviceImpl->rayTracingAccelerationStructureEnabled = selectedRayTracingAccelerationStructure;
    deviceImpl->capabilities.rayQuery = selectedRayQuery;
    deviceImpl->rayQueryEnabled = selectedRayQuery;
    deviceImpl->capabilities.pushDescriptor = selectedPushDescriptor;
    deviceImpl->pushDescriptorEnabled = selectedPushDescriptor;
    deviceImpl->bufferDeviceAddressEnabled =
        selectedBindlessDescriptorHeap || selectedRayTracingAccelerationStructure || selectedRayQuery;

    if (deviceImpl->debugUtilsEnabled) {
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
    deviceImpl->addQueue(graphicsQueue, deviceImpl->graphicsFamily, QueueType::Graphics);

    VkQueue computeQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(deviceImpl->device, deviceImpl->computeFamily, 0, &computeQueue);
    deviceImpl->addQueue(computeQueue, deviceImpl->computeFamily, QueueType::Compute);

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
            .format = toVkFormat(desc.format),
            .width = desc.width,
            .height = desc.height,
            .depth = desc.depth,
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

    std::cerr << label << " failed with Result " << resultToString(result) << '\n';
    return false;
}

constexpr const char* kTriangleShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kTriangleShaderModuleName = "triangle";
constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";
constexpr const char* kBindlessSmokeShaderModuleName = "bindless_smoke";
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
        std::cerr << "Slang compile failed for " << moduleName << "." << entryPointName << '\n';
        if (!compileResult.diagnostics.empty()) {
            std::cerr << compileResult.diagnostics << '\n';
        }
        return result;
    }
    if (!compileResult.diagnostics.empty()) {
        std::cerr << compileResult.diagnostics << '\n';
    }

    return device.createShaderModule(
        ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
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
        std::cerr << "SDL_Init failed: " << SDL_GetError() << '\n';
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
                    std::cerr << "Triangle preview pixel check failed: only "
                              << brightPixelCount << " bright pixels found.\n";
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
        std::cerr << "SDL_Init failed: " << SDL_GetError() << '\n';
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
                    std::cout << "VK_EXT_descriptor_heap unsupported; bindless smoke test skipped.\n";
                } else {
                    std::cerr << "createBindlessHeap was expected to return Unsupported, got "
                              << resultToString(result) << '\n';
                    exitCode = 1;
                }
            } else if (!checkResult(result, "createBindlessHeap")) {
                exitCode = resultToExitCode(result);
            } else {
                Queue* graphicsQueue = device->getQueue(QueueType::Graphics);
                if (graphicsQueue == nullptr) {
                    std::cerr << "No graphics queue available.\n";
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
                        std::cerr << "Failed to map bindless smoke readback buffer.\n";
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
                            std::cerr << "Bindless descriptor heap pixel check failed: "
                                      << matchedPixelCount << " matching pixels. First pixel RGBA=("
                                      << static_cast<uint32_t>(r) << ", "
                                      << static_cast<uint32_t>(g) << ", "
                                      << static_cast<uint32_t>(b) << ", "
                                      << static_cast<uint32_t>(a) << ").\n";
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
        std::cerr << "SDL_Init failed: " << SDL_GetError() << '\n';
        return 1;
    }

    const SDL_WindowFlags windowFlags =
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow("Metallic RHI Smoke Test", 1280, 720, windowFlags);
    if (window == nullptr) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << '\n';
        SDL_Quit();
        return 1;
    }

    std::unique_ptr<Device> device;
    std::unique_ptr<Swapchain> swapchain;
    std::vector<std::unique_ptr<TextureView>> swapchainViews;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Semaphore> imageAvailable;
    std::unique_ptr<Semaphore> renderFinished;
    std::unique_ptr<Fence> frameFence;

    auto cleanup = [&]() {
        if (device != nullptr) {
            (void)device->waitIdle();
        }

        swapchainViews.clear();
        commandBuffer.reset();
        commandPool.reset();
        imageAvailable.reset();
        renderFinished.reset();
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
        std::cerr << "SDL_GetWindowSizeInPixels failed: " << SDL_GetError() << '\n';
        cleanup();
        return 1;
    }

    Result result = createDevice(
        DeviceDesc{
            .applicationName = "Metallic RHI Smoke Test",
            .enableValidation = enableValidation,
        },
        device);
    if (!checkResult(result, "createDevice")) {
        cleanup();
        return resultToExitCode(result);
    }

    Queue* graphicsQueue = device->getQueue(QueueType::Graphics);
    if (graphicsQueue == nullptr) {
        std::cerr << "No graphics queue available.\n";
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

    if (!checkResult(device->createSemaphore(imageAvailable), "createSemaphore(imageAvailable)") ||
        !checkResult(device->createSemaphore(renderFinished), "createSemaphore(renderFinished)") ||
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
    SemaphoreSubmitDesc waitSemaphore{
        .semaphore = imageAvailable.get(),
        .stages = PipelineStageBits::ColorAttachment,
    };
    SemaphoreSubmitDesc signalSemaphore{
        .semaphore = renderFinished.get(),
        .stages = PipelineStageBits::AllCommands,
    };
    result = graphicsQueue->submit(QueueSubmitDesc{
        .waitSemaphores = &waitSemaphore,
        .waitSemaphoreCount = 1,
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalSemaphores = &signalSemaphore,
        .signalSemaphoreCount = 1,
        .signalFence = frameFence.get(),
    });
    if (!checkResult(result, "Queue::submit")) {
        cleanup();
        return resultToExitCode(result);
    }

    result = swapchain->present(*graphicsQueue, imageIndex, *renderFinished);
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
