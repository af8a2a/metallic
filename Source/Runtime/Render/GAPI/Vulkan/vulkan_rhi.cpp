#include "Runtime/Render/GAPI/rhi.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_VULKAN_VERSION 1004000
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <utility>
#include <vector>

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
        return Result::Success;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return Result::OutOfMemory;
    case VK_ERROR_DEVICE_LOST:
        return Result::DeviceLost;
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
        return Result::OutOfDate;
    case VK_ERROR_EXTENSION_NOT_PRESENT:
    case VK_ERROR_FEATURE_NOT_PRESENT:
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
    case VK_ERROR_INCOMPATIBLE_DRIVER:
    case VK_ERROR_LAYER_NOT_PRESENT:
        return Result::Unsupported;
    default:
        return Result::Failure;
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
    case Format::Bgra8Unorm:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case Format::Bgra8Srgb:
        return VK_FORMAT_B8G8R8A8_SRGB;
    case Format::Rgba8Unorm:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::Rgba8Srgb:
        return VK_FORMAT_R8G8B8A8_SRGB;
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
    case VK_FORMAT_B8G8R8A8_UNORM:
        return Format::Bgra8Unorm;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return Format::Bgra8Srgb;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return Format::Rgba8Unorm;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return Format::Rgba8Srgb;
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

struct CommandPoolImpl {
    DeviceImpl* device = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
};

struct CommandBufferImpl {
    DeviceImpl* device = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
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

struct DeviceImpl {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    uint32_t graphicsFamily = 0;
    bool sdlVulkanLoaded = false;
    bool validationEnabled = false;
    std::vector<std::unique_ptr<Queue>> queues;

    ~DeviceImpl();
    void addQueue(VkQueue queue, uint32_t familyIndex, QueueType type);
};

DeviceImpl::~DeviceImpl()
{
    queues.clear();

    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }

    if (allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
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
        SDL_Vulkan_UnloadLibrary();
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
        return Result::InvalidArgument;
    }

    auto* window = static_cast<SDL_Window*>(desc.window.nativeWindow);
    if (!SDL_Vulkan_CreateSurface(window, device->instance, nullptr, &surface)) {
        std::cerr << "SDL_Vulkan_CreateSurface failed: " << SDL_GetError() << '\n';
        return Result::Failure;
    }

    VkBool32 presentSupported = VK_FALSE;
    VkResult vkResult = vkGetPhysicalDeviceSurfaceSupportKHR(
        device->physicalDevice,
        device->graphicsFamily,
        surface,
        &presentSupported);
    if (vkResult != VK_SUCCESS || presentSupported == VK_FALSE) {
        return vkResult == VK_SUCCESS ? Result::Unsupported : resultFromVk(vkResult);
    }

    VkSurfaceCapabilitiesKHR capabilities{};
    vkResult = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device->physicalDevice, surface, &capabilities);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    uint32_t surfaceFormatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, surface, &surfaceFormatCount, nullptr);
    if (surfaceFormatCount == 0) {
        return Result::Unsupported;
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
    return Result::Success;
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
        return Result::InvalidArgument;
    }
    if ((desc.waitSemaphoreCount > 0 && desc.waitSemaphores == nullptr) ||
        (desc.commandBufferCount > 0 && desc.commandBuffers == nullptr) ||
        (desc.signalSemaphoreCount > 0 && desc.signalSemaphores == nullptr)) {
        return Result::InvalidArgument;
    }

    std::vector<VkSemaphoreSubmitInfo> waitSemaphores;
    waitSemaphores.reserve(desc.waitSemaphoreCount);
    for (uint32_t index = 0; index < desc.waitSemaphoreCount; ++index) {
        const SemaphoreSubmitDesc& wait = desc.waitSemaphores[index];
        if (wait.semaphore == nullptr || wait.semaphore->impl_ == nullptr) {
            return Result::InvalidArgument;
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
            return Result::InvalidArgument;
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
            return Result::InvalidArgument;
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
            return Result::InvalidArgument;
        }
        fence = desc.signalFence->impl_->fence;
    }

    return resultFromVk(vkQueueSubmit2(impl_->queue, 1, &submitInfo, fence));
}

Result Queue::waitIdle()
{
    if (impl_ == nullptr) {
        return Result::InvalidArgument;
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
        return Result::InvalidArgument;
    }

    const VkResult result = vkWaitForFences(
        impl_->device->device,
        1,
        &impl_->fence,
        VK_TRUE,
        timeoutNanoseconds);
    return result == VK_TIMEOUT ? Result::Failure : resultFromVk(result);
}

Result Fence::reset()
{
    if (impl_ == nullptr) {
        return Result::InvalidArgument;
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
        return Result::InvalidArgument;
    }

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    return resultFromVk(vkBeginCommandBuffer(impl_->commandBuffer, &beginInfo));
}

Result CommandBuffer::end()
{
    if (impl_ == nullptr) {
        return Result::InvalidArgument;
    }
    return resultFromVk(vkEndCommandBuffer(impl_->commandBuffer));
}

void CommandBuffer::barrier(const BarrierDesc& desc)
{
    if (impl_ == nullptr || desc.textureCount == 0 || desc.textures == nullptr) {
        return;
    }

    std::vector<VkImageMemoryBarrier2> imageBarriers;
    imageBarriers.reserve(desc.textureCount);

    for (uint32_t index = 0; index < desc.textureCount; ++index) {
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

    if (imageBarriers.empty()) {
        return;
    }

    VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    };
    vkCmdPipelineBarrier2(impl_->commandBuffer, &dependencyInfo);
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

    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {
            .offset = {desc.renderArea.x, desc.renderArea.y},
            .extent = {desc.renderArea.width, desc.renderArea.height},
        },
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
        .pColorAttachments = colorAttachments.data(),
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
        return Result::InvalidArgument;
    }
    return resultFromVk(vkResetCommandPool(impl_->device->device, impl_->pool, 0));
}

Result CommandPool::createCommandBuffer(std::unique_ptr<CommandBuffer>& outCommandBuffer)
{
    outCommandBuffer.reset();
    if (impl_ == nullptr) {
        return Result::InvalidArgument;
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
    return Result::Success;
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
        return Result::InvalidArgument;
    }

    const VkResult result = vkAcquireNextImageKHR(
        impl_->device->device,
        impl_->swapchain,
        kAcquireTimeoutNanoseconds,
        semaphore.impl_->semaphore,
        VK_NULL_HANDLE,
        &imageIndex);
    if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
        return Result::OutOfDate;
    }
    return resultFromVk(result);
}

Result Swapchain::present(Queue& queue, uint32_t imageIndex, Semaphore& waitSemaphore)
{
    if (impl_ == nullptr || queue.impl_ == nullptr || waitSemaphore.impl_ == nullptr) {
        return Result::InvalidArgument;
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
        return Result::OutOfDate;
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
        return Result::InvalidArgument;
    }
    return resultFromVk(vkDeviceWaitIdle(impl_->device));
}

Result Device::createSwapchain(const SwapchainDesc& desc, std::unique_ptr<Swapchain>& outSwapchain)
{
    outSwapchain.reset();
    if (impl_ == nullptr) {
        return Result::InvalidArgument;
    }

    auto swapchainImpl = std::make_unique<detail::SwapchainImpl>();
    swapchainImpl->device = impl_.get();
    const Result result = swapchainImpl->initialize(desc);
    if (result != Result::Success) {
        return result;
    }

    outSwapchain.reset(new Swapchain(std::move(swapchainImpl)));
    return Result::Success;
}

Result Device::createCommandPool(Queue& queue, std::unique_ptr<CommandPool>& outCommandPool)
{
    outCommandPool.reset();
    if (impl_ == nullptr || queue.impl_ == nullptr) {
        return Result::InvalidArgument;
    }

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
    return Result::Success;
}

Result Device::createFence(bool signaled, std::unique_ptr<Fence>& outFence)
{
    outFence.reset();
    if (impl_ == nullptr) {
        return Result::InvalidArgument;
    }

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
    return Result::Success;
}

Result Device::createSemaphore(std::unique_ptr<Semaphore>& outSemaphore)
{
    outSemaphore.reset();
    if (impl_ == nullptr) {
        return Result::InvalidArgument;
    }

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
    return Result::Success;
}

Result Device::createBuffer(const BufferDesc& desc, std::unique_ptr<Buffer>& outBuffer)
{
    outBuffer.reset();
    if (impl_ == nullptr || desc.size == 0) {
        return Result::InvalidArgument;
    }

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = desc.size,
        .usage = toVkBufferUsage(desc.usage),
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
    return Result::Success;
}

Result Device::createTexture(const TextureDesc& desc, std::unique_ptr<Texture>& outTexture)
{
    outTexture.reset();
    if (impl_ == nullptr || desc.format == Format::Unknown) {
        return Result::InvalidArgument;
    }

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
    return Result::Success;
}

Result Device::createTextureView(
    Texture& texture,
    const TextureViewDesc& desc,
    std::unique_ptr<TextureView>& outTextureView)
{
    outTextureView.reset();
    if (impl_ == nullptr || texture.impl_ == nullptr) {
        return Result::InvalidArgument;
    }

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
    return Result::Success;
}

Result createDevice(const DeviceDesc& desc, std::unique_ptr<Device>& outDevice)
{
    outDevice.reset();

    auto deviceImpl = std::make_unique<detail::DeviceImpl>();
    if (!SDL_Vulkan_LoadLibrary(nullptr)) {
        std::cerr << "SDL_Vulkan_LoadLibrary failed: " << SDL_GetError() << '\n';
        return Result::Unsupported;
    }
    deviceImpl->sdlVulkanLoaded = true;

    Uint32 sdlExtensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    if (sdlExtensions == nullptr || sdlExtensionCount == 0) {
        std::cerr << "SDL_Vulkan_GetInstanceExtensions failed: " << SDL_GetError() << '\n';
        return Result::Unsupported;
    }

    std::vector<const char*> instanceExtensions;
    instanceExtensions.reserve(sdlExtensionCount + 1);
    for (Uint32 index = 0; index < sdlExtensionCount; ++index) {
        instanceExtensions.push_back(sdlExtensions[index]);
    }

    const std::vector<VkExtensionProperties> availableExtensions = enumerateInstanceExtensions();
    const bool debugUtilsAvailable = hasName(availableExtensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (desc.enableValidation && debugUtilsAvailable) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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

    VkResult vkResult = vkCreateInstance(&instanceInfo, nullptr, &deviceImpl->instance);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    if (deviceImpl->validationEnabled && debugUtilsAvailable) {
        deviceImpl->debugMessenger = createDebugMessenger(deviceImpl->instance);
    }

    uint32_t physicalDeviceCount = 0;
    vkResult = vkEnumeratePhysicalDevices(deviceImpl->instance, &physicalDeviceCount, nullptr);
    if (vkResult != VK_SUCCESS || physicalDeviceCount == 0) {
        return vkResult == VK_SUCCESS ? Result::Unsupported : resultFromVk(vkResult);
    }

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(deviceImpl->instance, &physicalDeviceCount, physicalDevices.data());

    for (VkPhysicalDevice physicalDevice : physicalDevices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        if (properties.apiVersion < kVulkanApiVersion) {
            continue;
        }

        if (!hasDeviceExtension(physicalDevice, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            continue;
        }

        VkPhysicalDeviceVulkan13Features vulkan13Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        };
        VkPhysicalDeviceFeatures2 features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vulkan13Features,
        };
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
        if (vulkan13Features.dynamicRendering != VK_TRUE ||
            vulkan13Features.synchronization2 != VK_TRUE) {
            continue;
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        for (uint32_t queueIndex = 0; queueIndex < queueFamilyCount; ++queueIndex) {
            if ((queueFamilies[queueIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                continue;
            }

            deviceImpl->physicalDevice = physicalDevice;
            deviceImpl->graphicsFamily = queueIndex;
            break;
        }

        if (deviceImpl->physicalDevice != VK_NULL_HANDLE) {
            break;
        }
    }

    if (deviceImpl->physicalDevice == VK_NULL_HANDLE) {
        return Result::Unsupported;
    }

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = deviceImpl->graphicsFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    VkPhysicalDeviceVulkan13Features enabledVulkan13Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceFeatures2 enabledFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &enabledVulkan13Features,
    };

    const std::array<const char*, 1> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    VkDeviceCreateInfo deviceInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &enabledFeatures,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };

    vkResult = vkCreateDevice(deviceImpl->physicalDevice, &deviceInfo, nullptr, &deviceImpl->device);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = deviceImpl->physicalDevice;
    allocatorInfo.device = deviceImpl->device;
    allocatorInfo.instance = deviceImpl->instance;
    allocatorInfo.vulkanApiVersion = kVulkanApiVersion;
    vkResult = vmaCreateAllocator(&allocatorInfo, &deviceImpl->allocator);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(deviceImpl->device, deviceImpl->graphicsFamily, 0, &graphicsQueue);
    deviceImpl->addQueue(graphicsQueue, deviceImpl->graphicsFamily, QueueType::Graphics);

    outDevice.reset(new Device(std::move(deviceImpl)));
    return Result::Success;
}

namespace {

int resultToExitCode(Result result)
{
    return result == Result::Success ? 0 : 1;
}

bool checkResult(Result result, const char* label)
{
    if (result == Result::Success) {
        return true;
    }

    std::cerr << label << " failed with Result " << static_cast<int>(result) << '\n';
    return false;
}

} // namespace

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
            device->waitIdle();
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
