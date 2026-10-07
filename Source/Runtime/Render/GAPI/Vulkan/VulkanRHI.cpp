#include "VulkanTooling.h"
#include "Runtime/Render/GAPI/RHIEvents.h"
#include "VulkanSynchronization.h"
#include "VulkanTrace.h"
#include "VulkanResult.h"
#include "VulkanPipelineDiagnostics.h"
#include "VulkanValidation.h"
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/GAPI/QueueSubmissionIsolation.h"
#include "VulkanInterop.h"
#include "Runtime/Render/GAPI/TextureFormat.h"
#include "Runtime/Render/GAPI/PipelineCacheFile.h"
#include "Runtime/Render/GAPI/ShaderObjectCacheFile.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"
#include "Runtime/Render/GAPI/Hash.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanShaderPrintf.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "VulkanDeviceFeatures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSurfaceFormat.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanOpacityMicromap.h"
#include "Runtime/Render/GAPI/Vulkan/OpacityMicromapSPIRV.h"
#include "Runtime/Render/GAPI/Vulkan/OpacityMicromapBake.h"
#include "Runtime/Render/GAPI/Vulkan/DescriptorHeapShaderABI.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/GAPI/CommandSubmission.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
// winspool defines an ANSI/WIDE alias that collides with the RHI type.
#undef DeviceCapabilities
#endif

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
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
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

using detail::OpacityMicromapFormat;
using detail::OpacityMicromapTriangle;
using detail::OpacityMicromapUsage;

struct OpacityMicromapBuildInput {
    std::span<const OpacityMicromapUsage> usages;
    BufferSlice dataBuffer;
    BufferSlice triangleBuffer;
    uint64_t triangleStride = sizeof(OpacityMicromapTriangle);
};

constexpr uint32_t kVulkanAPIVersion = VK_API_VERSION_1_4;
constexpr uint64_t kAcquireTimeoutNanoseconds = std::numeric_limits<uint64_t>::max();
std::atomic_uint64_t nextResourceAllocationId{1};

ResourceMemoryInfo allocationMemoryInfo(const VmaAllocationInfo& allocation,
    const VkPhysicalDeviceMemoryProperties& properties, uint64_t allocationId)
{
    uint64_t blockId = 0;
    static_assert(sizeof(allocation.deviceMemory) <= sizeof(blockId));
    std::memcpy(&blockId, &allocation.deviceMemory, sizeof(allocation.deviceMemory));
    return {.allocationId = allocationId, .backingAllocationId = allocationId,
        .backingSizeBytes = allocation.size, .memoryBlockId = blockId,
        .offsetBytes = allocation.offset, .sizeBytes = allocation.size,
        .memoryTypeIndex = allocation.memoryType,
        .heapIndex = properties.memoryTypes[allocation.memoryType].heapIndex, .known = true};
}

// MTV3: native descriptor heaps and untyped pointers change the shader ABI.
// Reject cached pipelines from the previous mapped-array backend.
constexpr uint32_t kVulkanPipelineCacheBackendTag = 0x3356544du;

// Inspect the emitted interface, including shaders whose unused heaps were removed.
// Native heap runtime arrays have no DescriptorSet/Binding decorations.
bool spirvHasDescriptorBindings(const uint32_t* words, uint64_t byteSize)
{
    if (words == nullptr || byteSize < 5 * sizeof(uint32_t)) { return false; }
    const uint64_t size = byteSize / sizeof(uint32_t);
    for (uint64_t offset = 5; offset < size;) {
        const uint32_t count = words[offset] >> 16;
        if (count == 0 || count > size - offset) { return false; }
        if ((words[offset] & 0xffffu) == 71 && count >= 4 &&
            (words[offset + 2] == 33 || words[offset + 2] == 34)) { return true; }
        offset += count;
    }
    return false;
}

using vulkan::VulkanSyncScope;

using vulkan::resultFromVk;

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

// Volk loader/instance initialization still writes global loader state. Serialize
// creation only; normal device operations use immutable per-device dispatch.
std::mutex& volkInitializationMutex()
{
    static std::mutex mutex;
    return mutex;
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
    case Format::RG8Unorm:
        return VK_FORMAT_R8G8_UNORM;
    case Format::RG8Snorm:
        return VK_FORMAT_R8G8_SNORM;
    case Format::RG8Uint:
        return VK_FORMAT_R8G8_UINT;
    case Format::RG8Sint:
        return VK_FORMAT_R8G8_SINT;
    // LibNTC stores latent features as packed 4-bit BGRA channels.
    case Format::BGRA4Unorm:
        return VK_FORMAT_A4R4G4B4_UNORM_PACK16;
    case Format::BGRA8Unorm:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case Format::BGRA8sRGB:
        return VK_FORMAT_B8G8R8A8_SRGB;
    case Format::RGBA8Unorm:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case Format::BC4Unorm: return VK_FORMAT_BC4_UNORM_BLOCK;
    case Format::BC5Unorm: return VK_FORMAT_BC5_UNORM_BLOCK;
    case Format::BC7Unorm: return VK_FORMAT_BC7_UNORM_BLOCK;
    case Format::BC7sRGB: return VK_FORMAT_BC7_SRGB_BLOCK;
    case Format::RGBA8Snorm:
        return VK_FORMAT_R8G8B8A8_SNORM;
    case Format::RGBA8sRGB:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case Format::RGBA8Uint:
        return VK_FORMAT_R8G8B8A8_UINT;
    case Format::RGBA8Sint:
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
    case Format::RG16Unorm:
        return VK_FORMAT_R16G16_UNORM;
    case Format::RG16Snorm:
        return VK_FORMAT_R16G16_SNORM;
    case Format::RG16Uint:
        return VK_FORMAT_R16G16_UINT;
    case Format::RG16Sint:
        return VK_FORMAT_R16G16_SINT;
    case Format::RG16Sfloat:
        return VK_FORMAT_R16G16_SFLOAT;
    case Format::RGBA16Unorm:
        return VK_FORMAT_R16G16B16A16_UNORM;
    case Format::RGBA16Snorm:
        return VK_FORMAT_R16G16B16A16_SNORM;
    case Format::RGBA16Uint:
        return VK_FORMAT_R16G16B16A16_UINT;
    case Format::RGBA16Sint:
        return VK_FORMAT_R16G16B16A16_SINT;
    case Format::RGBA16Sfloat:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Format::R32Uint:
        return VK_FORMAT_R32_UINT;
    case Format::R32Sint:
        return VK_FORMAT_R32_SINT;
    case Format::R32Sfloat:
        return VK_FORMAT_R32_SFLOAT;
    case Format::RG32Uint:
        return VK_FORMAT_R32G32_UINT;
    case Format::RG32Sint:
        return VK_FORMAT_R32G32_SINT;
    case Format::RG32Sfloat:
        return VK_FORMAT_R32G32_SFLOAT;
    case Format::RGB32Uint:
        return VK_FORMAT_R32G32B32_UINT;
    case Format::RGB32Sint:
        return VK_FORMAT_R32G32B32_SINT;
    case Format::RGB32Sfloat:
        return VK_FORMAT_R32G32B32_SFLOAT;
    case Format::RGBA32Uint:
        return VK_FORMAT_R32G32B32A32_UINT;
    case Format::RGBA32Sint:
        return VK_FORMAT_R32G32B32A32_SINT;
    case Format::RGBA32Sfloat:
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
        return Format::RG8Unorm;
    case VK_FORMAT_R8G8_SNORM:
        return Format::RG8Snorm;
    case VK_FORMAT_R8G8_UINT:
        return Format::RG8Uint;
    case VK_FORMAT_R8G8_SINT:
        return Format::RG8Sint;
    case VK_FORMAT_A4R4G4B4_UNORM_PACK16:
        return Format::BGRA4Unorm;
    case VK_FORMAT_B8G8R8A8_UNORM:
        return Format::BGRA8Unorm;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return Format::BGRA8sRGB;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return Format::RGBA8Unorm;
    case VK_FORMAT_BC4_UNORM_BLOCK: return Format::BC4Unorm;
    case VK_FORMAT_BC5_UNORM_BLOCK: return Format::BC5Unorm;
    case VK_FORMAT_BC7_UNORM_BLOCK: return Format::BC7Unorm;
    case VK_FORMAT_BC7_SRGB_BLOCK: return Format::BC7sRGB;
    case VK_FORMAT_R8G8B8A8_SNORM:
        return Format::RGBA8Snorm;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return Format::RGBA8sRGB;
    case VK_FORMAT_R8G8B8A8_UINT:
        return Format::RGBA8Uint;
    case VK_FORMAT_R8G8B8A8_SINT:
        return Format::RGBA8Sint;
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
        return Format::RG16Unorm;
    case VK_FORMAT_R16G16_SNORM:
        return Format::RG16Snorm;
    case VK_FORMAT_R16G16_UINT:
        return Format::RG16Uint;
    case VK_FORMAT_R16G16_SINT:
        return Format::RG16Sint;
    case VK_FORMAT_R16G16_SFLOAT:
        return Format::RG16Sfloat;
    case VK_FORMAT_R16G16B16A16_UNORM:
        return Format::RGBA16Unorm;
    case VK_FORMAT_R16G16B16A16_SNORM:
        return Format::RGBA16Snorm;
    case VK_FORMAT_R16G16B16A16_UINT:
        return Format::RGBA16Uint;
    case VK_FORMAT_R16G16B16A16_SINT:
        return Format::RGBA16Sint;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return Format::RGBA16Sfloat;
    case VK_FORMAT_R32_UINT:
        return Format::R32Uint;
    case VK_FORMAT_R32_SINT:
        return Format::R32Sint;
    case VK_FORMAT_R32_SFLOAT:
        return Format::R32Sfloat;
    case VK_FORMAT_R32G32_UINT:
        return Format::RG32Uint;
    case VK_FORMAT_R32G32_SINT:
        return Format::RG32Sint;
    case VK_FORMAT_R32G32_SFLOAT:
        return Format::RG32Sfloat;
    case VK_FORMAT_R32G32B32_UINT:
        return Format::RGB32Uint;
    case VK_FORMAT_R32G32B32_SINT:
        return Format::RGB32Sint;
    case VK_FORMAT_R32G32B32_SFLOAT:
        return Format::RGB32Sfloat;
    case VK_FORMAT_R32G32B32A32_UINT:
        return Format::RGBA32Uint;
    case VK_FORMAT_R32G32B32A32_SINT:
        return Format::RGBA32Sint;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return Format::RGBA32Sfloat;
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

VkBufferUsageFlags2 toVkBufferUsage(BufferUsageBits usage)
{
    VkBufferUsageFlags2 flags = 0;
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
    if (hasFlag(usage, BufferUsageBits::MemoryDecompression)) {
        flags |= VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
    }
    return flags != 0 ? flags : VK_BUFFER_USAGE_TRANSFER_DST_BIT;
}

VkAddressCommandFlagsKHR addressCommandFlags(BufferUsageBits usage, bool aliased)
{
    // Address flags describe every VkBuffer overlapping the physical range,
    // not just the member targeted by this command. Shared allocations permit
    // mixed Storage usage; UNKNOWN covers every occupant and retained member.
    // Ordinary VMA buffers remain fully bound with their known storage usage.
    return VK_ADDRESS_COMMAND_FULLY_BOUND_BIT_KHR |
        (aliased ? VK_ADDRESS_COMMAND_UNKNOWN_STORAGE_BUFFER_USAGE_BIT_KHR :
            hasFlag(usage, BufferUsageBits::Storage) ? VK_ADDRESS_COMMAND_STORAGE_BUFFER_USAGE_BIT_KHR : 0);
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

Result<> makeOpacityMicromapGeometry(
    const VkPhysicalDeviceOpacityMicromapPropertiesKHR& properties,
    bool enabled,
    bool useExt,
    const OpacityMicromapBuildInput* input,
    RayTracingAccelerationStructureBuildFlags flags,
    bool requireBuffers,
    std::vector<VkMicromapUsageKHR>& usages,
    VkAccelerationStructureGeometryMicromapDataKHR& data,
    VkAccelerationStructureGeometryKHR& geometry)
{
    if (!enabled) {
        return makeError(Error::Unsupported);
    }
    if (input == nullptr || input->usages.empty() || input->usages.size() > UINT32_MAX ||
        input->triangleStride < sizeof(OpacityMicromapTriangle) || input->triangleStride % 4 != 0 ||
        hasFlag(flags, RayTracingAccelerationStructureBuildFlags::AllowUpdate) ||
        hasFlag(flags, RayTracingAccelerationStructureBuildFlags::AllowDataAccess)) {
        return makeError(Error::InvalidArgument);
    }
    if (useExt && hasFlag(flags, RayTracingAccelerationStructureBuildFlags::AllowCompaction)) {
        // The public compaction query pool holds AS queries, not EXT micromap queries.
        return makeError(Error::Unsupported);
    }
    uint64_t count = 0;
    usages.reserve(input->usages.size());
    for (uint32_t i = 0; i < input->usages.size(); ++i) {
        const OpacityMicromapUsage& usage = input->usages[i];
        if ((usage.format != OpacityMicromapFormat::TwoState && usage.format != OpacityMicromapFormat::FourState) ||
            usage.count == 0 || usage.subdivisionLevel > (usage.format == OpacityMicromapFormat::TwoState
                ? properties.maxOpacity2StateSubdivisionLevel : properties.maxOpacity4StateSubdivisionLevel)) {
            return makeError(Error::InvalidArgument);
        }
        count += usage.count;
        usages.push_back({usage.count, usage.subdivisionLevel, static_cast<VkOpacityMicromapFormatKHR>(usage.format)});
    }
    if (count > properties.maxMicromapTriangles) {
        return makeError(Error::InvalidArgument);
    }
    data = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR,
        .usageCountsCount = static_cast<uint32_t>(usages.size()), .pUsageCounts = usages.data(),
        .triangleArrayStride = input->triangleStride};
    if (requireBuffers) {
        const auto owner = input->dataBuffer.deviceIdentity();
        if (!input->dataBuffer.validate(owner, BufferUsageBits::AccelerationStructureBuildInput, 128) ||
            !input->triangleBuffer.validate(owner, BufferUsageBits::AccelerationStructureBuildInput, 128) ||
            count > input->triangleBuffer.size() / input->triangleStride) {
            return makeError(Error::InvalidArgument);
        }
        data.data = input->dataBuffer.deviceAddress();
        data.triangleArray = input->triangleBuffer.deviceAddress();
    }
    geometry = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .pNext = &data, .geometryType = VK_GEOMETRY_TYPE_MICROMAP_KHR};
    return {};
}

VkMicromapBuildInfoEXT makeExtMicromapBuildInfo(
    const VkAccelerationStructureGeometryMicromapDataKHR& data,
    RayTracingAccelerationStructureBuildFlags flags,
    std::vector<VkMicromapUsageEXT>& usages)
{
    usages.reserve(data.usageCountsCount);
    for (uint32_t i = 0; i < data.usageCountsCount; ++i) {
        const auto& usage = data.pUsageCounts[i];
        usages.push_back({usage.count, usage.subdivisionLevel, static_cast<uint32_t>(usage.format)});
    }
    VkBuildMicromapFlagsEXT extFlags = 0;
    if (hasFlag(flags, RayTracingAccelerationStructureBuildFlags::PreferFastTrace)) {
        extFlags |= VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
    }
    if (hasFlag(flags, RayTracingAccelerationStructureBuildFlags::PreferFastBuild)) {
        extFlags |= VK_BUILD_MICROMAP_PREFER_FAST_BUILD_BIT_EXT;
    }
    return {
        .sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT,
        .type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT,
        .flags = extFlags,
        .mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT,
        .usageCountsCount = static_cast<uint32_t>(usages.size()),
        .pUsageCounts = usages.data(),
        .data = {.deviceAddress = data.data},
        .triangleArray = {.deviceAddress = data.triangleArray},
        .triangleArrayStride = data.triangleArrayStride,
    };
}

Result<> makeExtMicromapAttachment(
    uint32_t primitiveCount,
    std::span<const OpacityMicromapUsage> sourceUsages,
    VkMicromapEXT micromap,
    std::vector<VkMicromapUsageEXT>& usages,
    VkAccelerationStructureTrianglesOpacityMicromapEXT& attachment,
    VkDeviceAddress identityIndexAddress = 0)
{
    if (micromap == VK_NULL_HANDLE || sourceUsages.empty() ||
        sourceUsages.size() > UINT32_MAX) {
        return makeError(Error::InvalidArgument);
    }
    uint64_t triangleCount = 0;
    usages.reserve(sourceUsages.size());
    for (uint32_t i = 0; i < sourceUsages.size(); ++i) {
        const auto& usage = sourceUsages[i];
        if (usage.count == 0 || (usage.format != OpacityMicromapFormat::TwoState &&
            usage.format != OpacityMicromapFormat::FourState)) {
            return makeError(Error::InvalidArgument);
        }
        triangleCount += usage.count;
        usages.push_back({usage.count, usage.subdivisionLevel, static_cast<uint32_t>(usage.format)});
    }
    if (triangleCount != primitiveCount) {
        return makeError(Error::InvalidArgument);
    }
    attachment = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT,
        // Size queries use the same index type/stride as builds, without reading
        // the address. Builds supply the retained identity buffer below.
        .indexType = VK_INDEX_TYPE_UINT32,
        .indexBuffer = {.deviceAddress = identityIndexAddress},
        .indexStride = sizeof(uint32_t),
        .usageCountsCount = static_cast<uint32_t>(usages.size()),
        .pUsageCounts = usages.data(),
        .micromap = micromap,
    };
    return {};
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
    if (hasFlag(flags, RayTracingAccelerationStructureBuildFlags::AllowDataAccess)) {
        result |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_BIT_KHR;
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


bool fillBufferImageLayout(
    const BufferTextureRegion& desc,
    uint32_t& outBufferRowLength,
    uint32_t& outBufferImageHeight)
{
    outBufferRowLength = 0;
    outBufferImageHeight = 0;
    const Format format = desc.texture->desc().format;
    const auto info = formatInfo(format);
    const auto footprint = textureCopyFootprint(format, desc.width, desc.height,
        uint64_t(desc.depth) * desc.layerCount, desc.bufferRowPitch, desc.bufferSlicePitch);
    if (!footprint || footprint->requiredBytes > desc.buffer.size() ||
        footprint->rowPitch % info.bytesPerBlock || footprint->slicePitch % footprint->rowPitch) { return false; }
    const uint64_t rowLength = footprint->rowPitch / info.bytesPerBlock * info.blockExtent;
    const uint64_t imageHeight = footprint->slicePitch / footprint->rowPitch * info.blockExtent;
    if ((desc.bufferRowPitch && rowLength > UINT32_MAX) ||
        (desc.bufferSlicePitch && imageHeight > UINT32_MAX)) { return false; }
    outBufferRowLength = desc.bufferRowPitch ? uint32_t(rowLength) : 0;
    outBufferImageHeight = desc.bufferSlicePitch ? uint32_t(imageHeight) : 0;
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

using vulkan::imageLayout;
using vulkan::toVkPipelineStages;
using vulkan::scopeInfo;

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

struct DebugCallbackContext {
    ValidationSink validation;
    vulkan::ShaderPrintf* printf = nullptr;
};

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData)
{
    const auto* context = static_cast<const DebugCallbackContext*>(userData);
    if (context && context->printf) {
        context->printf->capture(severity, *callbackData);
        // Printf sessions use the bounded raw queue; decoding/logging happens after completion.
        return VK_FALSE;
    }
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0 ||
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        spdlog::warn("Vulkan validation: {}", callbackData->pMessage);
    }
    const auto* sink = context ? &context->validation : nullptr;
    if (sink && sink->callback) {
        try {
            std::vector<ValidationObject> objects(callbackData->objectCount);
            for (uint32_t i = 0; i < callbackData->objectCount; ++i) {
                objects[i] = {callbackData->pObjects[i].objectHandle, vulkan::validationObjectType(callbackData->pObjects[i].objectType), callbackData->pObjects[i].pObjectName};
            }
            sink->callback(sink->context, {vulkan::validationSeverity(severity), vulkan::validationCategory(type), callbackData->messageIdNumber,
                callbackData->pMessageIdName, callbackData->pMessage, objects});
        } catch (...) {
            sink->callback(sink->context, {ValidationSeverity::Error, vulkan::validationCategory(type), 0,
                "Metallic.ValidationCaptureFailure", "Could not retain validation objects", {}});
        }
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance, DebugCallbackContext* sink)
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
        .pUserData = sink,
    };

    if (sink->printf && sink->printf->options().subscribeInfo) {
        info.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
    }
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (create(instance, &info, nullptr, &messenger) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    return messenger;
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

class DescriptorHeapWriter {
public:
    const VolkDeviceTable& functions() const { return *functions_; }
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
            heapProperties.maxPushDataSize > 0;
    }

    VkResult initialize(const VkPhysicalDeviceDescriptorHeapPropertiesEXT& heapProperties, VkDevice device, const VolkDeviceTable& functions)
    {
        *this = {};
        if (device == VK_NULL_HANDLE) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        if (heapProperties.samplerDescriptorSize == 0 ||
            heapProperties.imageDescriptorSize == 0 ||
            heapProperties.bufferDescriptorSize == 0 ||
            heapProperties.maxPushDataSize == 0) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        if (!heapProperties.imageDescriptorAlignment || !heapProperties.bufferDescriptorAlignment ||
            !heapProperties.samplerDescriptorAlignment ||
            heapProperties.samplerDescriptorSize % heapProperties.samplerDescriptorAlignment) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        const VkDeviceSize resourceStride = std::max(
            alignUp(heapProperties.imageDescriptorSize, heapProperties.imageDescriptorAlignment),
            alignUp(heapProperties.bufferDescriptorSize, heapProperties.bufferDescriptorAlignment));
        if (resourceStride % heapProperties.imageDescriptorAlignment != 0 ||
            resourceStride % heapProperties.bufferDescriptorAlignment != 0) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        device_ = device;
        functions_ = &functions;
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
    VkDeviceSize imageShaderDescriptorSize() const { return alignUp(imageDescriptorSize_, imageDescriptorAlignment_); }
    VkDeviceSize bufferShaderDescriptorSize() const { return alignUp(bufferDescriptorSize_, bufferDescriptorAlignment_); }
    VkDeviceSize resourceDescriptorStride() const { return std::max(imageShaderDescriptorSize(), bufferShaderDescriptorSize()); }
    VkDeviceSize samplerHeapAlignment() const { return samplerHeapAlignment_; }
    VkDeviceSize resourceHeapAlignment() const { return resourceHeapAlignment_; }
    VkDeviceSize maxSamplerHeapSize() const { return maxSamplerHeapSize_; }
    VkDeviceSize maxResourceHeapSize() const { return maxResourceHeapSize_; }
    VkDeviceSize minSamplerHeapReservedRange() const { return minSamplerHeapReservedRange_; }
    VkDeviceSize minResourceHeapReservedRange() const { return minResourceHeapReservedRange_; }
    VkDeviceSize maxPushDataSize() const { return maxPushDataSize_; }

    VkDeviceSize samplerOffset(uint32_t index) const { return samplerDescriptorSize_ * index; }
    VkDeviceSize imageOffset(uint32_t index) const { return resourceDescriptorStride() * index; }
    VkDeviceSize bufferOffset(uint32_t index) const { return resourceDescriptorStride() * index; }

    VkDeviceSize appendSamplerDescriptors(VkDeviceSize& offset, uint32_t count) const
    {
        const VkDeviceSize start = alignUp(offset, samplerDescriptorAlignment_);
        offset = start + samplerDescriptorSize_ * count;
        return start;
    }

    VkDeviceSize appendImageDescriptors(VkDeviceSize& offset, uint32_t count) const
    {
        const VkDeviceSize start = offset; // Resource regions contain complete unified slots.
        offset = start + resourceDescriptorStride() * count;
        return start;
    }

    VkDeviceSize appendBufferDescriptors(VkDeviceSize& offset, uint32_t count) const
    {
        const VkDeviceSize start = offset;
        offset = start + resourceDescriptorStride() * count;
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
        return functions_->vkWriteSamplerDescriptorsEXT(device_, 1, &samplerCreateInfo, &dstRange);
    }

    VkResult writeSamplerDescriptors(
        uint32_t descriptorCount,
        const VkSamplerCreateInfo* samplerCreateInfos,
        const VkHostAddressRangeEXT* dstRanges) const
    {
        if (!initialized() || descriptorCount == 0 || samplerCreateInfos == nullptr || dstRanges == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return functions_->vkWriteSamplerDescriptorsEXT(
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
        return functions_->vkWriteResourceDescriptorsEXT(device_, 1, &resourceInfo, &dstRange);
    }

    VkResult writeResourceDescriptors(
        uint32_t descriptorCount,
        const VkResourceDescriptorInfoEXT* resourceInfos,
        const VkHostAddressRangeEXT* dstRanges) const
    {
        if (!initialized() || descriptorCount == 0 || resourceInfos == nullptr || dstRanges == nullptr) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return functions_->vkWriteResourceDescriptorsEXT(device_, descriptorCount, resourceInfos, dstRanges);
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
        return functions_->vkWriteResourceDescriptorsEXT(device_, 1, &resourceInfo, &dstRange);
    }

    VkResult writeAccelerationStructureDescriptor(
        VkDeviceAddress accelerationStructureAddress,
        VkDeviceSize accelerationStructureSize,
        void* dst) const
    {
        if (!initialized() || dst == nullptr || accelerationStructureAddress == 0) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        // Size is unused for acceleration-structure descriptors. A non-zero range
        // is extra-validated as a buffer span and trips VUID-11483 when it is not
        // an address from vkGetAccelerationStructureDeviceAddressKHR.
        (void)accelerationStructureSize;
        VkDeviceAddressRangeEXT addressRange{
            .address = accelerationStructureAddress,
            .size = 0,
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
        return functions_->vkWriteResourceDescriptorsEXT(device_, 1, &resourceInfo, &dstRange);
    }

    VkResult writePartitionedAccelerationStructureDescriptor(
        VkDeviceAddress accelerationStructureAddress,
        VkDeviceSize accelerationStructureSize,
        void* dst) const
    {
        if (!initialized() || dst == nullptr || accelerationStructureAddress == 0) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        (void)accelerationStructureSize;
        VkDeviceAddressRangeEXT addressRange{
            .address = accelerationStructureAddress,
            .size = 0,
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
        return functions_->vkWriteResourceDescriptorsEXT(device_, 1, &resourceInfo, &dstRange);
    }

private:
    const VolkDeviceTable* functions_ = nullptr;
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

using vulkan::negotiation::VulkanExtensionSet;
using vulkan::negotiation::VulkanDeviceFeatureRequest;
using vulkan::negotiation::VulkanDeviceFeatureProbe;
using vulkan::negotiation::VulkanDeviceFeatureSelection;
using vulkan::negotiation::VulkanEnabledFeatureChain;
using vulkan::negotiation::enabledDeviceExtensions;

VulkanExtensionSet queryDeviceExtensions(VkPhysicalDevice physicalDevice)
{
    return VulkanExtensionSet::from(enumerateDeviceExtensions(physicalDevice),
        vulkan::toolingHooks().captureInjected());
}

void configureAftermathDiagnostics(VulkanEnabledFeatureChain& chain, const VulkanDeviceFeatureSelection& selection)
{
#if defined(VK_NV_device_diagnostics_config)
    if (const char* shaderDebugInfo = std::getenv("METALLIC_AFTERMATH_SHADER_DEBUG_INFO");
        shaderDebugInfo != nullptr && std::strcmp(shaderDebugInfo, "0") == 0) {
        chain.diagnosticsConfigCreateInfo.flags &= ~VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV;
    }
    const bool nsightAftermath = selection.aftermath && vulkan::toolingHooks().captureInjected();
    // TODO(Nsight Aftermath): Restore automatic checkpoints after an updated
    // capture runtime passes the injected Sponza OMM/BLAS compaction test.
    // Nsight 2026.3.1 + driver 616.64 faults with Error_DMA_PageFault here;
    // shader debug info and resource tracking remain enabled. See
    // Documentation/NsightKhrOpacityMicromapInvestigation.md.
    bool automaticCheckpoints = !nsightAftermath;
    if (const char* checkpoints = std::getenv("METALLIC_AFTERMATH_AUTOMATIC_CHECKPOINTS")) {
        if (std::strcmp(checkpoints, "0") == 0) { automaticCheckpoints = false; }
        if (std::strcmp(checkpoints, "1") == 0) { automaticCheckpoints = true; }
    }
    if (!automaticCheckpoints) {
        chain.diagnosticsConfigCreateInfo.flags &= ~VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV;
    }
    if (nsightAftermath && !automaticCheckpoints) {
        spdlog::warn("[Vulkan] Aftermath automatic checkpoints disabled during Nsight Graphics injection "
            "to avoid the RTAS GPU page fault; crash dumps, resource tracking and shader debug info remain available");
    }
#endif
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
    VkResult initialize(const VkPhysicalDeviceDescriptorHeapPropertiesEXT& heapProperties, VkDevice device, const VolkDeviceTable& functions)
    {
        *this = {};
        const VkResult result = writer_.initialize(heapProperties, device, functions);
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
        maxImageCapacity_ = capacityFromBytes(resourceAvailable, writer_.resourceDescriptorStride());
        maxBufferCapacity_ = capacityFromBytes(resourceAvailable, writer_.resourceDescriptorStride());
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
        const VkDeviceSize size = writer_.resourceDescriptorStride();
        return size > 0 ? static_cast<uint32_t>(imageRegionStartBytes_ / size) : 0;
    }

    uint32_t bufferShaderIndexBase() const
    {
        const VkDeviceSize size = writer_.resourceDescriptorStride();
        return size > 0 ? static_cast<uint32_t>(bufferRegionStartBytes_ / size) : 0;
    }

    bool allocate(BindlessHandleKind kind, BindlessHandle& outHandle)
    {
        uint32_t slot = 0;
        uint32_t shaderIndexBase = 0;
        switch (kind) {
        case BindlessHandleKind::Sampler:
            if (!allocateSlot(maxSamplers_, nextSamplerSlot_, freeSamplerSlots_, slot)) { return false; }
            break;
        case BindlessHandleKind::SampledImage:
        case BindlessHandleKind::StorageImage:
            if (!allocateSlot(maxImages_, nextImageSlot_, freeImageSlots_, slot)) { return false; }
            shaderIndexBase = imageShaderIndexBase();
            break;
        case BindlessHandleKind::Buffer:
        case BindlessHandleKind::AccelerationStructure:
            if (!allocateSlot(maxBuffers_, nextBufferSlot_, freeBufferSlots_, slot)) { return false; }
            shaderIndexBase = bufferShaderIndexBase();
            break;
        default:
            return false;
        }
        outHandle = {.kind = kind, .index = slot, .shaderIndex = shaderIndexBase + slot};
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
        if (handle.kind != BindlessHandleKind::AccelerationStructure ||
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
            writer_.functions().vkCmdBindSamplerHeapEXT(commandBuffer, &samplerBind);
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
            writer_.functions().vkCmdBindResourceHeapEXT(commandBuffer, &resourceBind);
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
    std::shared_ptr<std::mutex> nativeMutex;
    DeviceImpl* device = nullptr;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = 0;
    VkQueueFlags queueFlags = 0;
    uint32_t timestampValidBits = 0;
    QueueType type = QueueType::Graphics;
};

struct FenceImpl {
    DeviceImpl* device = nullptr;
    VkFence fence = VK_NULL_HANDLE;
    ~FenceImpl();
};

struct TimestampQueryPoolImpl {
    DeviceImpl* device = nullptr;
    TimestampQueryPoolDesc desc;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    uint32_t timestampValidBits = 0;
    double timestampPeriodNanoseconds = 0.0;
    ~TimestampQueryPoolImpl();
};

struct RayTracingAccelerationStructureCompactionQueryPoolImpl {
    DeviceImpl* device = nullptr;
    RayTracingAccelerationStructureCompactionQueryPoolDesc desc;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    ~RayTracingAccelerationStructureCompactionQueryPoolImpl();
};

struct SemaphoreImpl {
    DeviceImpl* device = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    ~SemaphoreImpl();
};

struct SwapchainSemaphoreImpl {
    DeviceImpl* device = nullptr;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    ~SwapchainSemaphoreImpl();
};

struct MemoryBudgetState {
    std::mutex mutex;
    MemoryBudgetPolicy policy;
    std::array<MemoryDomainBudget, size_t(MemoryBudgetDomain::Count)> domains{};
    uint64_t reservedBytes = 0, deniedAllocations = 0;
};

struct AliasBufferAllocation {
    DeviceImpl* device = nullptr;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint64_t allocationId = nextResourceAllocationId.fetch_add(1, std::memory_order_relaxed);
    uint64_t sizeBytes = 0;
    MemoryBudgetDomain memoryDomain = MemoryBudgetDomain::Other;
    bool deviceLocal = false;
    ~AliasBufferAllocation();
};

struct BufferImpl {
    DeviceImpl* device = nullptr;
    ResourceMemoryInfo memoryInfo{.allocationId = nextResourceAllocationId.fetch_add(1, std::memory_order_relaxed)};
    BufferDesc desc;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VmaAllocation allocation = VK_NULL_HANDLE;
    std::shared_ptr<AliasBufferAllocation> aliasAllocation;
    void* mapped = nullptr;
    uint64_t allocationBytes = 0;
    bool deviceLocal = false;
    ~BufferImpl();
};

struct BufferAddressCommandAccess {
    static BufferImpl* allocation(const BufferSlice& slice) { return slice.allocation_.get(); }
    static VkAddressCommandFlagsKHR flags(const Buffer& buffer)
    {
        return flags(buffer.impl_.get());
    }

    static VkAddressCommandFlagsKHR flags(const BufferSlice& slice)
    {
        return flags(slice.allocation_.get());
    }

    static bool overlap(const BufferSlice& source, const BufferSlice& destination)
    {
        const auto& before = source.allocation_;
        const auto& after = destination.allocation_;
        const bool sameBacking = before == after ||
            (before && after && before->aliasAllocation && before->aliasAllocation == after->aliasAllocation);
        // Alias members bind at the same backing offset; slice offsets are
        // therefore also their physical offsets relative to that allocation.
        return sameBacking && source.offset_ < destination.offset_ + destination.size_ &&
            destination.offset_ < source.offset_ + source.size_;
    }

private:
    static VkAddressCommandFlagsKHR flags(const BufferImpl* allocation)
    {
        return addressCommandFlags(allocation ? allocation->desc.usage : BufferUsageBits::None,
            allocation && allocation->aliasAllocation);
    }
};

struct MicromapBufferAllocation {
    DeviceImpl* device = nullptr;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    uint32_t triangleCount = 0;
    uint64_t allocationBytes = 0;
    bool deviceLocal = false;
    MemoryBudgetDomain domain = MemoryBudgetDomain::RayTracing;
    uint64_t dataOffset = 0;
    uint64_t triangleOffset = 0;
    ~MicromapBufferAllocation();
};

struct PartitionedTopLevelState {
    PartitionedAccelerationStructureDesc desc;
};

struct RayTracingAccelerationStructureImpl {
    DeviceImpl* device = nullptr;
    RayTracingAccelerationStructureDesc desc;
    std::unique_ptr<Buffer> storage;
    VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
    VkMicromapEXT micromap = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    std::mutex micromapIndexMutex;
    std::vector<std::unique_ptr<MicromapBufferAllocation>> micromapIndexBuffers;
    std::vector<std::shared_ptr<RayTracingAccelerationStructureImpl>> coverageDependencies;
    std::shared_ptr<void> coverageBuildIdentity;
    RayTracingCoverageAccelerationStats coverageStats;

    std::unique_ptr<PartitionedTopLevelState> partitioned;

    ~RayTracingAccelerationStructureImpl();
};

struct PreparedGeometryCoverage {
    std::shared_ptr<RayTracingAccelerationStructureImpl> resource;
    BakedOpacityMicromap baked;
    RayTracingAccelerationStructureBuildSizes sizes;
};

struct RayTracingBottomLevelBuildPlanImpl {
    DeviceImpl* device = nullptr;
    RayTracingAccelerationStructureBuildFlags flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace;
    std::vector<RayTracingTriangleGeometryDesc> geometries;
    std::vector<PreparedGeometryCoverage> coverage;
    RayTracingAccelerationStructureBuildSizes sizes;
    RayTracingCoverageAccelerationStats coverageStats;
    std::shared_ptr<void> identity = std::make_shared<uint8_t>(0);
    std::atomic_bool recorded{false};
    std::shared_ptr<std::atomic_uint64_t> bakeBudget;
    uint64_t bakeBytes = 0;

    ~RayTracingBottomLevelBuildPlanImpl()
    {
        coverage.clear();
        if (bakeBudget && bakeBytes != 0) { bakeBudget->fetch_sub(bakeBytes, std::memory_order_relaxed); }
    }
};

struct BufferViewImpl {
    DeviceImpl* device = nullptr;
    std::shared_ptr<BufferImpl> buffer;
    BufferViewDesc desc;
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
};

struct AliasTextureAllocation {
    DeviceImpl* device = nullptr;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint64_t allocationId = nextResourceAllocationId.fetch_add(1, std::memory_order_relaxed);
    uint64_t sizeBytes = 0;
    bool deviceLocal = false;
    ~AliasTextureAllocation();
};

struct TextureImpl {
    DeviceImpl* device = nullptr;
    ResourceMemoryInfo memoryInfo{.allocationId = nextResourceAllocationId.fetch_add(1, std::memory_order_relaxed)};
    TextureDesc desc;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    std::shared_ptr<AliasTextureAllocation> aliasAllocation;
    VkImageCreateFlags flags = 0;
    VkImageUsageFlags usage = 0;
    bool ownsImage = false;
    uint64_t allocationSize = 0;
    bool deviceLocal = false;
    ~TextureImpl();
};

struct TextureViewImpl {
    DeviceImpl* device = nullptr;
    std::shared_ptr<TextureImpl> texture;
    TextureViewDesc desc;
    mutable std::mutex mutex;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageViewCreateInfo createInfo() const;
    Result<> materialize();
    ~TextureViewImpl();
};

struct ShaderModuleImpl {
    std::vector<uint32_t> deviceSpirv;
    bool hasDescriptorBindings = false;
    DeviceImpl* device = nullptr;
    VkShaderModule module = VK_NULL_HANDLE;
    uint64_t contentHash = 0;
    uint64_t inputSpirvFnv1a64 = 0;
    uint64_t deviceSpirvFnv1a64 = 0;
    std::vector<uint8_t> replayInputSpirv, replayDeviceSpirv;
    std::string diagnosticName;
    ~ShaderModuleImpl();
};

struct PipelineCacheImpl {
    DeviceImpl* device = nullptr;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::filesystem::path filePath;
    PipelineCacheFileIdentity fileIdentity;
    PipelineCacheStats stats;
    std::unordered_set<uint64_t> storedPsoHashes;
    std::unordered_set<uint64_t> sessionPsoHashes;
    mutable std::mutex mutex;
    std::mutex saveMutex;
    bool saveOnDestroy = true;

    ~PipelineCacheImpl();
    Result<> initialize(DeviceImpl& owningDevice, const PipelineCacheDesc& desc);
    Result<> save();
    bool recordPsoLocked(uint64_t psoHash);
};

struct GraphicsPipelineImpl {
    DeviceImpl* device = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool usesBindlessHeap = false;
    uint64_t psoHash = 0;
    bool pipelineCacheHit = false;
    ~GraphicsPipelineImpl();
};

struct ComputePipelineImpl {
    DeviceImpl* device = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool usesBindlessHeap = false;
    uint64_t psoHash = 0;
    bool pipelineCacheHit = false;
    std::vector<uint8_t> replayInputSpirv, replayDeviceSpirv;
    ~ComputePipelineImpl();
};

struct GraphicsShaderObjectProgramImpl {
    DeviceImpl* device = nullptr;
    VkShaderEXT vertexShader = VK_NULL_HANDLE;
    VkShaderEXT fragmentShader = VK_NULL_HANDLE;
    bool usesBindlessHeap = false;
    ShaderObjectCacheStats cacheStats;
    std::string binaryCacheFilePath;
    ~GraphicsShaderObjectProgramImpl();
};

struct CaptureCommandPool {
    VkCommandPool pool = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    std::vector<VkCommandBuffer> availableBuffers;
};

struct CommandPoolImpl {
    std::shared_ptr<CommandSubmissionRegistry> submissions = std::make_shared<CommandSubmissionRegistry>();
    DeviceImpl* device = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkQueueFlags queueFlags = 0;
    bool recycleForCapture = false;
    std::vector<VkCommandBuffer> availableBuffers;
    ~CommandPoolImpl();
};

struct CommandBufferImpl {
    SynchronizationStats synchronizationStats;
    std::shared_ptr<CommandSubmissionRegistry> submissions;
    DeviceImpl* device = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    CommandPoolImpl* capturePool = nullptr;
    uint32_t queueFamilyIndex = 0;
    VkQueueFlags queueFlags = 0;
    VkPipelineLayout currentGraphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout currentComputePipelineLayout = VK_NULL_HANDLE;
    VkPipeline currentComputePipeline = VK_NULL_HANDLE;
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
    DisplayOutputMode outputMode = DisplayOutputMode::SDR;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<std::unique_ptr<Texture>> textures;

    ~SwapchainImpl();
    Result<> initialize(const SwapchainDesc& desc);
    void wrapImages(const std::vector<VkImage>& images, TextureUsageBits usage);
};

struct BindlessHeapBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
    VkDeviceSize mappedOffset = 0;
    uint64_t allocationBytes = 0;
    bool deviceLocal = false;
};

struct BindlessHeapImpl {
    DeviceImpl* device = nullptr;
    BindlessHeapDesc desc;
    DescriptorHeap heap;
    BindlessHeapBuffer samplerHeap;
    BindlessHeapBuffer resourceHeap;

    ~BindlessHeapImpl();
    Result<> initialize(DeviceImpl& owningDevice, const BindlessHeapDesc& heapDesc);
    Result<> createHeapBuffer(VkDeviceSize size, VkDeviceSize alignment, BindlessHeapBuffer& outBuffer);
    void destroyHeapBuffer(BindlessHeapBuffer& buffer);
    void flushSamplerDirty();
    void flushResourceDirty();
};

struct DeviceImpl {
    std::mutex sharedStateMutex;
    // Requests for the same binary serialize creation/export; unrelated keys
    // may compile concurrently. Files are also replaced atomically on disk.
    std::array<std::mutex, 16> shaderObjectCacheMutexes;
    std::unordered_map<const void*, std::shared_ptr<void>> sharedStates;
    DebugCallbackContext debugContext;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VolkDeviceTable functions{};
    VolkInstanceTable instanceFunctions{};
    vulkan::VulkanDeviceProperties physicalProperties;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    VmaAllocator allocator = VK_NULL_HANDLE;
    std::array<VmaPool, VK_MAX_MEMORY_TYPES> materialImagePools{};
    std::shared_ptr<MemoryBudgetState> memoryBudgetState = std::make_shared<MemoryBudgetState>();
    bool memoryBudgetExtension = false;
    bool hdrMetadataExtension = false;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    mutable std::chrono::steady_clock::time_point lastBudgetRefresh{};
    mutable uint32_t budgetRefreshIndex = 0;
    DeviceMemoryBudget memoryBudgetLocked() const;
    bool admitMemoryLocked(uint32_t memoryType, uint64_t bytes, MemoryBudgetDomain domain);
    Result<> prepareBufferAllocationLocked(
        const VkBufferCreateInfo& info,
        VmaAllocationCreateInfo& allocationInfo,
        MemoryBudgetDomain domain);
    void trackMemoryLocked(MemoryBudgetDomain domain, uint64_t bytes, bool local, bool add);
    DeviceCapabilities capabilities;
    vulkan::VulkanDeviceCapabilities vulkanCapabilities;
    std::shared_ptr<std::atomic_uint64_t> coverageBakeBytes = std::make_shared<std::atomic_uint64_t>(0);
    PipelineCacheFileIdentity pipelineCacheFileIdentity;
    DescriptorHeapWriter descriptorHeapWriter;
    uint32_t graphicsFamily = 0;
    uint32_t computeFamily = 0;
    uint32_t copyFamily = UINT32_MAX;
    uint32_t copyQueueIndex = UINT32_MAX;
    uint32_t computeQueueIndex = 0;
    SDL_SharedObject* vulkanLoaderHandle = nullptr;
    bool sdlVulkanLoaded = false;
    bool validationEnabled = false;
    bool synchronizationValidationEnabled = false;
    bool debugUtilsEnabled = false;
    bool bindlessDescriptorHeapEnabled = false;
    bool shaderUntypedPointersEnabled = false;
    bool pipelineExecutableStatistics = false;
    bool logPipelineKeys = false;
    bool bufferDeviceAddressEnabled = false;
    bool rayTracingPipelineEnabled = false;
    bool opacityMicromapExt = false;
    bool streamlineInitialized = false;
    PFN_vkSetDebugUtilsObjectNameEXT setDebugUtilsObjectName = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT cmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmdEndDebugUtilsLabel = nullptr;
    PFN_vkGetCalibratedTimestampsEXT getCalibratedTimestamps = nullptr;
    VkTimeDomainEXT calibrationHostDomain = VK_TIME_DOMAIN_DEVICE_EXT;
    std::vector<std::unique_ptr<Queue>> queues;
    std::mutex captureCommandPoolMutex;
    std::vector<CaptureCommandPool> captureCommandPools;

    ~DeviceImpl();
    void addQueue(
        VkQueue queue,
        uint32_t familyIndex,
        VkQueueFlags queueFlags,
        uint32_t timestampValidBits,
        QueueType type);
};

DeviceMemoryBudget DeviceImpl::memoryBudgetLocked() const
{
    DeviceMemoryBudget result;
    const auto& state = *memoryBudgetState;
    result.policy = state.policy;
    result.reservedBytes = state.reservedBytes;
    result.deniedAllocations = state.deniedAllocations;
    result.domains = state.domains;
    result.driverBudget = memoryBudgetExtension;
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
    // VMA otherwise refreshes only after enough allocation operations; external
    // processes and DLSS can change pressure while our allocation count is idle.
    const auto now = std::chrono::steady_clock::now();
    if (memoryBudgetExtension && now - lastBudgetRefresh >= std::chrono::milliseconds(100)) {
        vmaSetCurrentFrameIndex(allocator, ++budgetRefreshIndex);
        lastBudgetRefresh = now;
    }
    vmaGetHeapBudgets(allocator, budgets);
    for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; ++i) {
        const auto& heap = memoryProperties.memoryHeaps[i];
        const bool local = (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        uint64_t limit = std::min<uint64_t>(heap.size, budgets[i].budget);
        if (local && state.policy.deviceLocalHeapLimitBytes) {
            limit = std::min(limit, state.policy.deviceLocalHeapLimitBytes);
        }
        result.heaps.push_back({heap.size, limit, std::max<uint64_t>(budgets[i].usage, budgets[i].statistics.blockBytes),
            budgets[i].statistics.blockBytes, budgets[i].statistics.allocationBytes,
            budgets[i].statistics.blockCount, budgets[i].statistics.allocationCount, local});
        if (local && (result.primaryDeviceLocalHeap == UINT32_MAX ||
                heap.size > memoryProperties.memoryHeaps[result.primaryDeviceLocalHeap].size)) {
            result.primaryDeviceLocalHeap = i;
        }
    }
    if (result.primaryDeviceLocalHeap != UINT32_MAX) {
        const auto& heap = result.heaps[result.primaryDeviceLocalHeap];
        uint64_t available = heap.budgetBytes - std::min(heap.budgetBytes, heap.usageBytes);
        if (state.policy.enabled) {
            available -= std::min(available, state.policy.safetyBytes);
            available -= std::min(available, state.reservedBytes);
        }
        result.availableBytes = available;
    }
    return result;
}

bool DeviceImpl::admitMemoryLocked(uint32_t memoryType, uint64_t bytes, MemoryBudgetDomain domain)
{
    auto& state = *memoryBudgetState;
    if (!state.policy.enabled) { return true; }
    const uint32_t heapIndex = memoryProperties.memoryTypes[memoryType].heapIndex;
    if (!(memoryProperties.memoryHeaps[heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) { return true; }
    const auto budget = memoryBudgetLocked();
    const auto& heap = budget.heaps[heapIndex];
    uint64_t available = heap.budgetBytes - std::min(heap.budgetBytes, heap.usageBytes);
    available -= std::min(available, state.policy.safetyBytes);
    available -= std::min(available, state.reservedBytes);
    if (bytes <= available) { return true; }
    ++state.deniedAllocations;
    spdlog::error("[GPUBudget] Allocation denied domain={} requested={} available={} heap={} usage={} budget={} reserved={} safety={}",
        uint32_t(domain), bytes, available, heapIndex, heap.usageBytes, heap.budgetBytes,
        state.reservedBytes, state.policy.safetyBytes);
    spdlog::error("[GPUBudget] Denied allocation context blockBytes={} allocationBytes={} textures={} uploadLocal={}",
        heap.blockBytes, heap.allocationBytes,
        state.domains[size_t(MemoryBudgetDomain::MaterialTextures)].deviceLocalBytes,
        state.domains[size_t(MemoryBudgetDomain::Upload)].deviceLocalBytes);
    return false;
}

void DeviceImpl::trackMemoryLocked(MemoryBudgetDomain domain, uint64_t bytes, bool local, bool add)
{
    auto& stats = memoryBudgetState->domains[size_t(domain)];
    if (add) {
        stats.allocationBytes += bytes;
        if (local) { stats.deviceLocalBytes += bytes; }
        ++stats.allocationCount;
        stats.peakAllocationBytes = std::max(stats.peakAllocationBytes, stats.allocationBytes);
    } else {
        stats.allocationBytes -= bytes;
        if (local) { stats.deviceLocalBytes -= bytes; }
        --stats.allocationCount;
    }
}

Result<> DeviceImpl::prepareBufferAllocationLocked(
    const VkBufferCreateInfo& info,
    VmaAllocationCreateInfo& allocationInfo,
    MemoryBudgetDomain domain)
{
    uint32_t memoryType = 0;
    const VkResult result = vmaFindMemoryTypeIndexForBufferInfo(allocator, &info, &allocationInfo, &memoryType);
    if (result != VK_SUCCESS) { return resultFromVk(result); }
    VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    const VkDeviceBufferMemoryRequirements request{.sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS, .pCreateInfo = &info};
    functions.vkGetDeviceBufferMemoryRequirements(device, &request, &requirements);
    if (!admitMemoryLocked(memoryType, requirements.memoryRequirements.size, domain)) { return makeError(Error::OutOfMemory); }
    allocationInfo.memoryTypeBits = 1u << memoryType;
    if (memoryBudgetState->policy.enabled) { allocationInfo.flags |= VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT; }
    return {};
}

// Keep native ownership in Impl so unique_ptr replacement and wrapper
// destruction perform the same cleanup.
FenceImpl::~FenceImpl()
{
    if (fence != VK_NULL_HANDLE) {

        device->functions.vkDestroyFence(device->device, fence, nullptr);
        fence = VK_NULL_HANDLE;
    }
}

TimestampQueryPoolImpl::~TimestampQueryPoolImpl()
{
    if (queryPool != VK_NULL_HANDLE) {

        device->functions.vkDestroyQueryPool(device->device, queryPool, nullptr);
        queryPool = VK_NULL_HANDLE;
    }
}

RayTracingAccelerationStructureCompactionQueryPoolImpl::~RayTracingAccelerationStructureCompactionQueryPoolImpl()
{
    if (queryPool != VK_NULL_HANDLE) {

        device->functions.vkDestroyQueryPool(device->device, queryPool, nullptr);
        queryPool = VK_NULL_HANDLE;
    }
}

SemaphoreImpl::~SemaphoreImpl()
{
    if (semaphore != VK_NULL_HANDLE) {

        vulkan::forgetTraceObject(device->device, VK_OBJECT_TYPE_SEMAPHORE, uint64_t(semaphore));
        device->functions.vkDestroySemaphore(device->device, semaphore, nullptr);
        semaphore = VK_NULL_HANDLE;
    }
}

SwapchainSemaphoreImpl::~SwapchainSemaphoreImpl()
{
    if (semaphore != VK_NULL_HANDLE) {

        vulkan::forgetTraceObject(device->device, VK_OBJECT_TYPE_SEMAPHORE, uint64_t(semaphore));
        device->functions.vkDestroySemaphore(device->device, semaphore, nullptr);
        semaphore = VK_NULL_HANDLE;
    }
}

ShaderModuleImpl::~ShaderModuleImpl()
{
    if (module != VK_NULL_HANDLE) {

        device->functions.vkDestroyShaderModule(device->device, module, nullptr);
        module = VK_NULL_HANDLE;
    }
}

CommandPoolImpl::~CommandPoolImpl()
{
    if (pool != VK_NULL_HANDLE) {

        if (recycleForCapture) {
            (void)device->functions.vkResetCommandPool(device->device, pool, 0);
            std::lock_guard lock(device->captureCommandPoolMutex);
            device->captureCommandPools.push_back({
                pool, queueFamilyIndex, std::move(availableBuffers)});
        } else {
            device->functions.vkDestroyCommandPool(device->device, pool, nullptr);
        }
        pool = VK_NULL_HANDLE;
        submissions->cancel();
    }
}

MicromapBufferAllocation::~MicromapBufferAllocation()
{
    if (!device || buffer == VK_NULL_HANDLE) { return; }
    std::lock_guard lock(device->memoryBudgetState->mutex);
    vmaDestroyBuffer(allocator, buffer, allocation);
    device->trackMemoryLocked(domain, allocationBytes, deviceLocal, false);
}

BufferImpl::~BufferImpl()
{
    if (!device || buffer == VK_NULL_HANDLE) { return; }
    vulkan::forgetTraceObject(device->device, VK_OBJECT_TYPE_BUFFER, uint64_t(buffer));
    // Alias members and temporary unbound buffers own only their native handle.
    // The last retained member releases and accounts for the backing allocation.
    if (aliasAllocation || allocation == VK_NULL_HANDLE) {
        device->functions.vkDestroyBuffer(device->device, buffer, nullptr);
        return;
    }
    std::lock_guard lock(device->memoryBudgetState->mutex);
    if (mapped) { vmaUnmapMemory(device->allocator, allocation); }
    vmaDestroyBuffer(device->allocator, buffer, allocation);
    device->trackMemoryLocked(desc.memoryDomain, allocationBytes, deviceLocal, false);
}

AliasBufferAllocation::~AliasBufferAllocation()
{
    if (!device || allocation == VK_NULL_HANDLE) { return; }
    std::lock_guard lock(device->memoryBudgetState->mutex);
    vmaFreeMemory(device->allocator, allocation);
    device->trackMemoryLocked(memoryDomain, sizeBytes, deviceLocal, false);
}

AliasTextureAllocation::~AliasTextureAllocation()
{
    if (!device || allocation == VK_NULL_HANDLE) { return; }
    std::lock_guard lock(device->memoryBudgetState->mutex);
    vmaFreeMemory(device->allocator, allocation);
    device->trackMemoryLocked(MemoryBudgetDomain::FrameResources, sizeBytes, deviceLocal, false);
}

TextureImpl::~TextureImpl()
{
    if (!device || image == VK_NULL_HANDLE) { return; }
    vulkan::forgetTraceObject(device->device, VK_OBJECT_TYPE_IMAGE, uint64_t(image));
    if (!ownsImage) { return; }
    // Aliased images retain the allocation owner; unbound images are temporary
    // members of a group being created. Neither frees memory or tracks it here.
    if (aliasAllocation || allocation == VK_NULL_HANDLE) {
        device->functions.vkDestroyImage(device->device, image, nullptr);
        return;
    }
    std::lock_guard lock(device->memoryBudgetState->mutex);
    vmaDestroyImage(device->allocator, image, allocation);
    device->trackMemoryLocked(desc.memoryDomain, allocationSize, deviceLocal, false);
}

VkImageViewCreateInfo TextureViewImpl::createInfo() const
{
    auto type = toVkImageViewType(texture->desc.type);
    if (desc.range.layerCount > 1) {
        if (type == VK_IMAGE_VIEW_TYPE_1D) { type = VK_IMAGE_VIEW_TYPE_1D_ARRAY; }
        if (type == VK_IMAGE_VIEW_TYPE_2D) { type = VK_IMAGE_VIEW_TYPE_2D_ARRAY; }
    }
    return {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = texture->image,
        .viewType = type, .format = format,
        .components = {static_cast<VkComponentSwizzle>(desc.swizzle[0]),
            static_cast<VkComponentSwizzle>(desc.swizzle[1]), static_cast<VkComponentSwizzle>(desc.swizzle[2]),
            static_cast<VkComponentSwizzle>(desc.swizzle[3])},
        .subresourceRange = {aspectForFormat(desc.format), desc.range.baseMip, desc.range.mipCount, desc.range.baseLayer, desc.range.layerCount}};
}

Result<> TextureViewImpl::materialize()
{
    std::lock_guard lock(mutex);
    if (view != VK_NULL_HANDLE) { return {}; }
    const auto info = createInfo();

    return resultFromVk(device->functions.vkCreateImageView(device->device, &info, nullptr, &view));
}

TextureViewImpl::~TextureViewImpl()
{
    if (device && view != VK_NULL_HANDLE) {
        device->functions.vkDestroyImageView(device->device, view, nullptr);
    }
}

RayTracingAccelerationStructureImpl::~RayTracingAccelerationStructureImpl()
{
    if (device != nullptr && micromap != VK_NULL_HANDLE) {

        device->functions.vkDestroyMicromapEXT(device->device, micromap, nullptr);
        micromap = VK_NULL_HANDLE;
    }
    if (device != nullptr && accelerationStructure != VK_NULL_HANDLE) {

        device->functions.vkDestroyAccelerationStructureKHR(device->device, accelerationStructure, nullptr);
        accelerationStructure = VK_NULL_HANDLE;
    }
}

DeviceImpl::~DeviceImpl()
{
    queues.clear();

    if (device != VK_NULL_HANDLE) {

        const VkResult waitResult = functions.vkDeviceWaitIdle(device);
        if (waitResult == VK_ERROR_DEVICE_LOST) {
            vulkan::toolingHooks().deviceLost();
        }
    }

    if (streamlineInitialized) {
        vulkan::shutdownStreamline();
        streamlineInitialized = false;
    }

    sharedStates.clear();

    if (allocator != VK_NULL_HANDLE) {
        for (auto pool : materialImagePools) {
            if (pool) { vmaDestroyPool(allocator, pool); }
        }
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE) {

        for (const auto& pool : captureCommandPools) {
            functions.vkDestroyCommandPool(device, pool.pool, nullptr);
        }
        captureCommandPools.clear();
        functions.vkDestroyDevice(device, nullptr);

        device = VK_NULL_HANDLE;
    }

    if (debugMessenger != VK_NULL_HANDLE) {
        instanceFunctions.vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        debugMessenger = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        instanceFunctions.vkDestroyInstance(instance, nullptr);
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

    // Registry's device-owned writer has already joined before cache handles
    // are destroyed. Explicit cache owners also finish all accesses first.
    if (saveOnDestroy && !filePath.empty()) {
        const Result<> result = save();
        if (!result) {
            spdlog::warn("Failed to save pipeline cache '{}'", filePath.string());
        }
    }
    device->functions.vkDestroyPipelineCache(device->device, pipelineCache, nullptr);
    pipelineCache = VK_NULL_HANDLE;
}

Result<> PipelineCacheImpl::initialize(DeviceImpl& owningDevice, const PipelineCacheDesc& desc)
{
    device = &owningDevice;
    saveOnDestroy = desc.saveOnDestroy;
    fileIdentity = owningDevice.pipelineCacheFileIdentity;
    if (desc.filePath != nullptr && desc.filePath[0] != '\0') {
        filePath = desc.filePath;
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
    VkResult vkResult = device->functions.vkCreatePipelineCache(
        owningDevice.device,
        &createInfo,
        nullptr,
        &pipelineCache);
    if (vkResult != VK_SUCCESS && createInfo.initialDataSize > 0) {
        stats.loadStatus = PipelineCacheLoadStatus::Incompatible;
        fileData = {};
        createInfo.initialDataSize = 0;
        createInfo.pInitialData = nullptr;
        vkResult = device->functions.vkCreatePipelineCache(
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

Result<> PipelineCacheImpl::save()
{
    // Serialize file replacement without holding the PSO creation/statistics
    // mutex during driver extraction, checksum calculation or durable I/O.
    std::lock_guard saveLock(saveMutex);
    if (device == nullptr || pipelineCache == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }
    if (filePath.empty()) {
        return {};
    }
    const auto begin = std::chrono::steady_clock::now();
    uint64_t revision = 0;
    std::vector<uint64_t> hashes;
    {
        std::lock_guard lock(mutex);
        if (stats.dirtyRevision == stats.persistedRevision) {
            return {};
        }
        revision = stats.dirtyRevision;
        hashes.reserve(storedPsoHashes.size() + sessionPsoHashes.size());
        hashes.insert(hashes.end(), storedPsoHashes.begin(), storedPsoHashes.end());
        hashes.insert(hashes.end(), sessionPsoHashes.begin(), sessionPsoHashes.end());
        stats.saveInProgress = true;
    }

    struct SaveAttempt {
        PipelineCacheImpl& cache;
        bool completed = false;
        ~SaveAttempt()
        {
            if (!completed) {
                std::lock_guard lock(cache.mutex);
                ++cache.stats.saveFailureCount;
                cache.stats.saveInProgress = false;
            }
        }
    } attempt{*this};
    const auto nanoseconds = [](auto duration) {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
    };
    const auto extractBegin = std::chrono::steady_clock::now();
    // Cache flags are zero: Vulkan internally synchronizes native cache access.
    // Snapshot hashes BEFORE extraction so they never describe a later PSO
    // that is missing from the extracted blob. Extra native entries are safe.
    size_t byteSize = 0;
    VkResult vkResult = device->functions.vkGetPipelineCacheData(
        device->device,
        pipelineCache,
        &byteSize,
        nullptr);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    std::vector<uint8_t> backendData(byteSize);
    constexpr uint32_t kMaxExtractAttempts = 3;
    for (uint32_t retry = 0; retry < kMaxExtractAttempts; ++retry) {
        size_t writtenSize = backendData.size();
        vkResult = device->functions.vkGetPipelineCacheData(
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
        if (retry + 1 == kMaxExtractAttempts) {
            // A successful size query is not a successful data extraction.
            // Leave the revision pending if concurrent cache growth exhausted
            // retries instead of persisting a truncated/unfilled buffer.
            return resultFromVk(VK_INCOMPLETE);
        }

        byteSize = 0;
        vkResult = device->functions.vkGetPipelineCacheData(
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

    const auto extractEnd = std::chrono::steady_clock::now();
    const auto writeBegin = extractEnd;
    std::string reason;
    if (!savePipelineCacheFile(filePath, fileIdentity, hashes, backendData, reason)) {
        spdlog::warn("Failed to write pipeline cache '{}': {}", filePath.string(), reason);
        return makeError(Error::Failure);
    }

    const auto writeEnd = std::chrono::steady_clock::now();
    uint64_t storedCount = 0;
    const uint64_t extractNs = nanoseconds(extractEnd - extractBegin);
    const uint64_t writeNs = nanoseconds(writeEnd - writeBegin);
    uint64_t saveNs = 0;
    {
        std::lock_guard lock(mutex);
        storedPsoHashes.insert(hashes.begin(), hashes.end());
        storedCount = stats.storedPsoCount = storedPsoHashes.size();
        stats.backendDataSize = backendData.size();
        stats.persistedRevision = revision;
        ++stats.saveCount;
        stats.lastExtractTimeNanoseconds = extractNs;
        stats.lastWriteTimeNanoseconds = writeNs;
        saveNs = stats.lastSaveTimeNanoseconds = nanoseconds(std::chrono::steady_clock::now() - begin);
        stats.saveInProgress = false;
        attempt.completed = true;
    }
    spdlog::info(
        "Saved pipeline cache '{}' with {} PSO hashes and {} backend bytes (revision={}, extractMs={:.3f}, writeMs={:.3f}, totalMs={:.3f})",
        filePath.string(),
        storedCount,
        backendData.size(),
        revision,
        static_cast<double>(extractNs) / 1'000'000.0,
        static_cast<double>(writeNs) / 1'000'000.0,
        static_cast<double>(saveNs) / 1'000'000.0);
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
        ++stats.dirtyRevision;
    }
    return cacheHit;
}

void DeviceImpl::addQueue(
    VkQueue queue,
    uint32_t familyIndex,
    VkQueueFlags queueFlags,
    uint32_t timestampValidBits,
    QueueType type)
{
    auto impl = std::make_unique<QueueImpl>();
    impl->device = this;
    impl->queue = queue;
    for (const auto& existing : queues) {
        if (existing->impl_->queue == queue) { impl->nativeMutex = existing->impl_->nativeMutex; break; }
    }
    if (!impl->nativeMutex) { impl->nativeMutex = std::make_shared<std::mutex>(); }
    impl->familyIndex = familyIndex;
    impl->queueFlags = queueFlags;
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

Result<> ensureMicromapIdentityIndices(
    RayTracingAccelerationStructureImpl& micromap,
    uint32_t triangleCount,
    VkDeviceAddress& address)
{
    address = 0;
    if (triangleCount == 0 || micromap.micromap == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }
    // TODO(Nsight OMM replay): Remove explicit identity indices once Nsight
    // restores VK_INDEX_TYPE_NONE_KHR attachments correctly. 2026.3.1 reads a
    // null index-buffer record in both ngfx-replay and ngfx-rpc. See
    // Documentation/NsightCaptureReplayInvestigation.md for removal criteria.
    // EXT is selected only for capture injection. Keep each immutable allocation
    // with the OMM, including older capacities used by in-flight BLAS builds.
    std::scoped_lock lock(micromap.micromapIndexMutex);
    for (const auto& indices : micromap.micromapIndexBuffers) {
        if (indices->triangleCount >= triangleCount) {
            address = indices->address;
            return {};
        }
    }
    DeviceImpl& device = *micromap.device;
    const auto families = queueFamiliesForAccess(device, QueueAccessBits::Graphics | QueueAccessBits::Compute);
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = uint64_t(triangleCount) * sizeof(uint32_t),
        .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT,
        .sharingMode = families.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = families.size() > 1 ? static_cast<uint32_t>(families.size()) : 0,
        .pQueueFamilyIndices = families.size() > 1 ? families.data() : nullptr,
    };
    auto allocationInfo = allocationInfoForMemory(MemoryLocation::HostUpload);
    auto indices = std::make_unique<MicromapBufferAllocation>();
    std::unique_lock budgetLock(device.memoryBudgetState->mutex);
    const Result<> admitted = device.prepareBufferAllocationLocked(bufferInfo, allocationInfo, MemoryBudgetDomain::RayTracing);
    if (!admitted) { return admitted; }
    indices->device = &device;
    indices->allocator = device.allocator;
    VmaAllocationInfo allocatedInfo{};
    VkResult result = vmaCreateBuffer(device.allocator, &bufferInfo, &allocationInfo,
        &indices->buffer, &indices->allocation, &allocatedInfo);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    indices->allocationBytes = allocatedInfo.size;
    indices->deviceLocal = (device.memoryProperties.memoryHeaps[device.memoryProperties.memoryTypes[allocatedInfo.memoryType].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    device.trackMemoryLocked(MemoryBudgetDomain::RayTracing, indices->allocationBytes, indices->deviceLocal, true);
    budgetLock.unlock();
    void* mapped = nullptr;
    result = vmaMapMemory(device.allocator, indices->allocation, &mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    auto* values = static_cast<uint32_t*>(mapped);
    for (uint32_t i = 0; i < triangleCount; ++i) {
        values[i] = i;
    }
    result = vmaFlushAllocation(device.allocator, indices->allocation, 0, bufferInfo.size);
    vmaUnmapMemory(device.allocator, indices->allocation);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    // Flushed host writes precede submission of the recorded BLAS build; the
    // queue submit performs the host-to-device domain operation.
    const VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = indices->buffer,
    };
    indices->address = device.functions.vkGetBufferDeviceAddress(device.device, &addressInfo);
    if (indices->address == 0) {
        return makeError(Error::Failure);
    }
    indices->triangleCount = triangleCount;
    address = indices->address;
    micromap.micromapIndexBuffers.push_back(std::move(indices));
    return {};
}

BindlessHeapImpl::~BindlessHeapImpl()
{
    destroyHeapBuffer(samplerHeap);
    destroyHeapBuffer(resourceHeap);
}

Result<> BindlessHeapImpl::initialize(DeviceImpl& owningDevice, const BindlessHeapDesc& heapDesc)
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

    VkResult vkResult = heap.initialize(device->physicalProperties.descriptorHeap, device->device, device->functions);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    if (desc.maxSamplers > 0) {
        if (heap.setupSamplerHeap(desc.maxSamplers) == 0) {
            return makeError(Error::Unsupported);
        }
        Result<> result = createHeapBuffer(heap.samplerHeapSize(), heap.samplerHeapAlignment(), samplerHeap);
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

Result<> BindlessHeapImpl::createHeapBuffer(VkDeviceSize size, VkDeviceSize alignment, BindlessHeapBuffer& outBuffer)
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
    std::unique_lock budgetLock(device->memoryBudgetState->mutex);
    const Result<> admitted = device->prepareBufferAllocationLocked(bufferInfo, allocationInfo, MemoryBudgetDomain::FrameResources);
    if (!admitted) { return admitted; }
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
    const VkDeviceAddress rawAddress = device->functions.vkGetBufferDeviceAddress(device->device, &addressInfo);
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
        .allocationBytes = allocatedInfo.size,
        .deviceLocal = (device->memoryProperties.memoryHeaps[device->memoryProperties.memoryTypes[allocatedInfo.memoryType].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0,
    };
    device->trackMemoryLocked(MemoryBudgetDomain::FrameResources, outBuffer.allocationBytes, outBuffer.deviceLocal, true);
    return {};
}

void BindlessHeapImpl::destroyHeapBuffer(BindlessHeapBuffer& buffer)
{
    if (device != nullptr && buffer.buffer != VK_NULL_HANDLE) {
        std::lock_guard lock(device->memoryBudgetState->mutex);
        vmaDestroyBuffer(device->allocator, buffer.buffer, buffer.allocation);
        device->trackMemoryLocked(MemoryBudgetDomain::FrameResources, buffer.allocationBytes, buffer.deviceLocal, false);
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
        device->functions.vkDestroySwapchainKHR(device->device, swapchain, nullptr);
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

Result<> SwapchainImpl::initialize(const SwapchainDesc& desc)
{
    if (desc.window.system != WindowSystem::SDL3 || desc.window.nativeWindow == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    auto* window = static_cast<SDL_Window*>(desc.window.nativeWindow);
    if (!SDL_Vulkan_CreateSurface(window, device->instance, nullptr, &surface)) {
        spdlog::error("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
        return makeError(Error::Failure);
    }

    VkBool32 presentSupported = VK_FALSE;
    VkResult vkResult = device->instanceFunctions.vkGetPhysicalDeviceSurfaceSupportKHR(
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
    vkResult = device->instanceFunctions.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device->physicalDevice, surface, &capabilities);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    uint32_t surfaceFormatCount = 0;
    vkResult = device->instanceFunctions.vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, surface, &surfaceFormatCount, nullptr);
    if (vkResult != VK_SUCCESS) { return resultFromVk(vkResult); }
    if (surfaceFormatCount == 0) {
        return makeError(Error::Unsupported);
    }
    std::vector<VkSurfaceFormatKHR> surfaceFormats(surfaceFormatCount);
    vkResult = device->instanceFunctions.vkGetPhysicalDeviceSurfaceFormatsKHR(
        device->physicalDevice,
        surface,
        &surfaceFormatCount,
        surfaceFormats.data());
    if (vkResult != VK_SUCCESS) { return resultFromVk(vkResult); }
    surfaceFormats.resize(surfaceFormatCount);
    VkSurfaceFormatKHR selectedFormat{};
    if (!vulkan::selectSurfaceFormat(surfaceFormats, toVkFormat(desc.format), desc.outputMode,
            desc.allowSdrFallback, selectedFormat, outputMode)) {
        spdlog::error("No supported surface format for requested display output mode");
        return makeError(Error::Unsupported);
    }
    if (outputMode != desc.outputMode) {
        spdlog::warn("{} surface pair unavailable; falling back to SDR_sRGB", displayOutputName(desc.outputMode));
    }
    spdlog::info("Swapchain output: {}, VkFormat {}, VkColorSpace {}",
        displayOutputName(outputMode),
        static_cast<int>(selectedFormat.format), static_cast<int>(selectedFormat.colorSpace));

    uint32_t presentModeCount = 0;
    device->instanceFunctions.vkGetPhysicalDeviceSurfacePresentModesKHR(device->physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    if (presentModeCount > 0) {
        device->instanceFunctions.vkGetPhysicalDeviceSurfacePresentModesKHR(
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

    vkResult = device->functions.vkCreateSwapchainKHR(device->device, &createInfo, nullptr, &swapchain);
    if (vkResult != VK_SUCCESS) {
        return resultFromVk(vkResult);
    }

    if (outputMode == DisplayOutputMode::HDR10_PQ && device->hdrMetadataExtension && device->functions.vkSetHdrMetadataEXT) {
        const float peak = std::isfinite(desc.peakNits) ? std::clamp(desc.peakNits, 80.0f, 10000.0f) : 1000.0f;
        VkHdrMetadataEXT metadata{.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
            .displayPrimaryRed = {0.708f, 0.292f}, .displayPrimaryGreen = {0.170f, 0.797f},
            .displayPrimaryBlue = {0.131f, 0.046f}, .whitePoint = {0.3127f, 0.3290f},
            .maxLuminance = peak, .minLuminance = 0.0f,
            .maxContentLightLevel = 0.0f, .maxFrameAverageLightLevel = 0.0f};
        // Content light levels are unknown until measured, not guessed from mastering peak.
        device->functions.vkSetHdrMetadataEXT(device->device, 1, &swapchain, &metadata);
    }

    uint32_t actualImageCount = 0;
    device->functions.vkGetSwapchainImagesKHR(device->device, swapchain, &actualImageCount, nullptr);
    std::vector<VkImage> images(actualImageCount);
    device->functions.vkGetSwapchainImagesKHR(device->device, swapchain, &actualImageCount, images.data());

    vkFormat = selectedFormat.format;
    format = fromVkFormat(selectedFormat.format);
    width = extent.width;
    height = extent.height;
    wrapImages(images, textureUsage);
    spdlog::info("[Swapchain] presentMode={} vsync={} images={} extent={}x{}",
        static_cast<uint32_t>(presentMode), desc.vsync, actualImageCount, width, height);
    return {};
}

} // namespace detail

METALLIC_RHI_HANDLE_DEFINITIONS(Queue)

namespace {
vulkan::SyncSupport syncSupport(const detail::DeviceImpl& device, VkQueueFlags queues)
{
    return {queues, device.capabilities.rayTracingAccelerationStructure,
        device.capabilities.memoryDecompression, device.rayTracingPipelineEnabled,
        device.capabilities.bindlessDescriptorHeap, device.capabilities.cooperativeVector};
}
} // namespace

Result<> Queue::submit(const QueueSubmitDesc& desc)
{
    return submitImpl(desc, false);
}

Result<> Queue::submitTracked(const QueueSubmitDesc& desc)
{
    return submitImpl(desc, true);
}

Result<> Queue::submitImpl(const QueueSubmitDesc& desc, bool tracked)
{
    const detail::QueueSubmissionAccess submissionAccess;
    if (impl_ == nullptr || impl_->queue == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }
    if ((desc.waitSemaphores.size() > UINT32_MAX) ||
        (desc.waitSwapchainSemaphores.size() > UINT32_MAX) ||
        (desc.commandBuffers.size() > UINT32_MAX) ||
        (desc.signalSemaphores.size() > UINT32_MAX) ||
        (desc.signalSwapchainSemaphores.size() > UINT32_MAX)) {
        return makeError(Error::InvalidArgument);
    }

    const RHIOperationScope submitMarker(RHIOperation::Submit, desc.commandBuffers.size());

    const auto support = syncSupport(*impl_->device, impl_->queueFlags);
    std::vector<VkSemaphoreSubmitInfo> waitSemaphores;
    waitSemaphores.reserve(desc.waitSemaphores.size() + desc.waitSwapchainSemaphores.size());
    for (uint32_t index = 0; index < desc.waitSemaphores.size(); ++index) {
        const SemaphoreSubmitDesc& wait = desc.waitSemaphores[index];
        if (wait.semaphore == nullptr || wait.semaphore->impl_ == nullptr ||
            !vulkan::validDeviceStages(wait.stages, support)) {
            return makeError(Error::InvalidArgument);
        }
        waitSemaphores.push_back({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = wait.semaphore->impl_->semaphore,
            .value = wait.value,
            .stageMask = toVkPipelineStages(wait.stages),
        });
    }
    for (uint32_t index = 0; index < desc.waitSwapchainSemaphores.size(); ++index) {
        const SwapchainSemaphoreSubmitDesc& wait = desc.waitSwapchainSemaphores[index];
        if (wait.semaphore == nullptr || wait.semaphore->impl_ == nullptr ||
            !vulkan::validDeviceStages(wait.stages, support)) {
            return makeError(Error::InvalidArgument);
        }
        waitSemaphores.push_back({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = wait.semaphore->impl_->semaphore,
            .stageMask = toVkPipelineStages(wait.stages),
        });
    }

    std::vector<VkCommandBufferSubmitInfo> commandBuffers;
    std::unordered_map<CommandSubmissionContext*, size_t> retentionCounts;
    commandBuffers.reserve(desc.commandBuffers.size());
    for (uint32_t index = 0; index < desc.commandBuffers.size(); ++index) {
        CommandBuffer* commandBuffer = desc.commandBuffers[index];
        if (commandBuffer == nullptr || commandBuffer->impl_ == nullptr ||
            commandBuffer->submission_ == nullptr || !commandBuffer->submission_->finished.load(std::memory_order_acquire) ||
            !commandBuffer->submission_->canSubmit() || commandBuffer->recording_ ||
            (commandBuffer->submission_->sealed && !tracked)) {
            return makeError(Error::InvalidArgument);
        }
        if (auto* context = commandBuffer->submissionContext_.get()) {
            if (!context->canSubmit(tracked, commandBuffer->submission_->sealed)) { return makeError(Error::InvalidArgument); }
            retentionCounts[context] += commandBuffer->submission_->resources.size();
        }
        for (const auto& wait : commandBuffer->dependencyWaits_) {
            if (!wait.semaphore || !wait.semaphore->impl_) { return makeError(Error::InvalidArgument); }
            const VkSemaphore semaphore = wait.semaphore->impl_->semaphore;
            const auto existing = std::find_if(waitSemaphores.begin(), waitSemaphores.end(),
                [&](const auto& entry) { return entry.semaphore == semaphore; });
            if (existing == waitSemaphores.end()) {
                waitSemaphores.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .semaphore = semaphore, .value = wait.value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
            } else {
                existing->value = std::max(existing->value, wait.value);
                existing->stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            }
        }
        commandBuffers.push_back({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = commandBuffer->impl_->commandBuffer,
        });
    }

    std::vector<VkSemaphoreSubmitInfo> signalSemaphores;
    signalSemaphores.reserve(desc.signalSemaphores.size() + desc.signalSwapchainSemaphores.size());
    for (uint32_t index = 0; index < desc.signalSemaphores.size(); ++index) {
        const SemaphoreSubmitDesc& signal = desc.signalSemaphores[index];
        if (signal.semaphore == nullptr || signal.semaphore->impl_ == nullptr ||
            !vulkan::validDeviceStages(signal.stages, support)) {
            return makeError(Error::InvalidArgument);
        }
        signalSemaphores.push_back({
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = signal.semaphore->impl_->semaphore,
            .value = signal.value,
            .stageMask = toVkPipelineStages(signal.stages),
        });
    }
    for (uint32_t index = 0; index < desc.signalSwapchainSemaphores.size(); ++index) {
        const SwapchainSemaphoreSubmitDesc& signal = desc.signalSwapchainSemaphores[index];
        if (signal.semaphore == nullptr || signal.semaphore->impl_ == nullptr ||
            !vulkan::validDeviceStages(signal.stages, support)) {
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

    // Allocate before queue acceptance. The successful ownership handoff below
    // cannot allocate or release GPU resources, including after partial failure.
    for (const auto& [context, count] : retentionCounts) {
        context->reserveResources(count);
    }
    const Result<> result = resultFromVk(vulkan::submitInterop(*this, {&submitInfo, 1}, fence));
    if (result) {
        // Mark the whole accepted batch before invoking any CPU publication hooks.
        for (uint32_t index = 0; index < desc.commandBuffers.size(); ++index) {
            auto& commands = *desc.commandBuffers[index];
            commands.submission_->submitted = true;
            if (auto* context = commands.submissionContext_.get()) {
                context->acceptResources(commands.submission_->resources);
            }
        }
        for (uint32_t index = 0; index < desc.commandBuffers.size(); ++index) {
            desc.commandBuffers[index]->submission_->submit();
        }
    }
    return result;
}

bool Queue::sameQueue(const Queue& other) const
{
    return impl_ && other.impl_ && impl_->device == other.impl_->device && impl_->queue == other.impl_->queue;
}

Result<> Queue::waitIdle()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromVk(vulkan::waitInterop(*this));
}

QueueType Queue::type() const
{
    return impl_ != nullptr ? impl_->type : QueueType::Graphics;
}

uint32_t Queue::timestampValidBits() const
{
    return impl_ != nullptr ? impl_->timestampValidBits : 0;
}

Result<GPUClockCalibration> Queue::calibrateTimestamps() const
{
    GPUClockCalibration calibration{};
    if (impl_ == nullptr) { return makeError(Error::InvalidArgument); }
    const auto& device = *impl_->device;
    if (impl_->timestampValidBits == 0 || device.getCalibratedTimestamps == nullptr ||
        device.calibrationHostDomain == VK_TIME_DOMAIN_DEVICE_EXT) {
        return makeError(Error::Unsupported);
    }
    const VkCalibratedTimestampInfoEXT info[] = {
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_DEVICE_EXT},
        {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, device.calibrationHostDomain},
    };
    uint64_t timestamps[2]{};
    uint64_t deviation = 0;
    const VkResult result = device.getCalibratedTimestamps(device.device, 2, info, timestamps, &deviation);
    if (result != VK_SUCCESS) { return std::unexpected(resultFromVk(result).error()); }
    uint64_t cpuNanoseconds = timestamps[1];
#if defined(_WIN32)
    // Divide before multiplying so long-running QPC counters cannot overflow.
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    const uint64_t ticksPerSecond = static_cast<uint64_t>(frequency.QuadPart);
    cpuNanoseconds = (timestamps[1] / ticksPerSecond) * 1'000'000'000ull +
        static_cast<uint64_t>(static_cast<double>(timestamps[1] % ticksPerSecond) *
            1'000'000'000.0 / static_cast<double>(ticksPerSecond));
#endif
    calibration = {timestamps[0], cpuNanoseconds, deviation};
    return calibration;
}

METALLIC_RHI_HANDLE_DEFINITIONS(Fence)

Result<> Fence::wait(uint64_t timeoutNanoseconds)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    const RHIOperationScope waitMarker(RHIOperation::FenceWait, timeoutNanoseconds);

    const VkResult result = impl_->device->functions.vkWaitForFences(
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

Result<> Fence::reset()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromVk(impl_->device->functions.vkResetFences(impl_->device->device, 1, &impl_->fence));
}

bool Fence::isSignaled() const
{
    return impl_ != nullptr &&
        impl_->device->functions.vkGetFenceStatus(impl_->device->device, impl_->fence) == VK_SUCCESS;
}

METALLIC_RHI_HANDLE_DEFINITIONS(TimestampQueryPool)

const TimestampQueryPoolDesc& TimestampQueryPool::desc() const
{
    static const TimestampQueryPoolDesc kEmptyDesc;
    return impl_ != nullptr ? impl_->desc : kEmptyDesc;
}

Result<> TimestampQueryPool::reset(uint32_t firstQuery, uint32_t queryCount)
{
    if (impl_ == nullptr || queryCount == 0 ||
        firstQuery >= impl_->desc.queryCount ||
        queryCount > impl_->desc.queryCount - firstQuery) {
        return makeError(Error::InvalidArgument);
    }

    impl_->device->functions.vkResetQueryPool(impl_->device->device, impl_->queryPool, firstQuery, queryCount);
    return {};
}

Result<> TimestampQueryPool::readResults(
    uint32_t firstQuery,
    std::span<TimestampQueryResult> outResults) const
{
    if (impl_ == nullptr ||
        outResults.size() > UINT32_MAX ||
        outResults.size() == 0 ||
        firstQuery >= impl_->desc.queryCount ||
        outResults.size() > impl_->desc.queryCount - firstQuery) {
        return makeError(Error::InvalidArgument);
    }

    struct RawTimestampQueryResult {
        uint64_t value = 0;
        uint64_t available = 0;
    };
    std::vector<RawTimestampQueryResult> rawResults(outResults.size());

    const VkResult result = impl_->device->functions.vkGetQueryPoolResults(
        impl_->device->device,
        impl_->queryPool,
        firstQuery,
        static_cast<uint32_t>(outResults.size()),
        rawResults.size() * sizeof(RawTimestampQueryResult),
        rawResults.data(),
        sizeof(RawTimestampQueryResult),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS && result != VK_NOT_READY) {
        return resultFromVk(result);
    }

    for (uint32_t index = 0; index < outResults.size(); ++index) {
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

METALLIC_RHI_HANDLE_DEFINITIONS(RayTracingAccelerationStructureCompactionQueryPool)

const RayTracingAccelerationStructureCompactionQueryPoolDesc&
RayTracingAccelerationStructureCompactionQueryPool::desc() const
{
    static const RayTracingAccelerationStructureCompactionQueryPoolDesc kEmptyDesc;
    return impl_ != nullptr ? impl_->desc : kEmptyDesc;
}

Result<> RayTracingAccelerationStructureCompactionQueryPool::readResults(
    uint32_t firstQuery,
    std::span<uint64_t> outCompactedSizes) const
{
    if (impl_ == nullptr ||
        outCompactedSizes.size() > UINT32_MAX ||
        outCompactedSizes.size() == 0 ||
        firstQuery >= impl_->desc.queryCount ||
        outCompactedSizes.size() > impl_->desc.queryCount - firstQuery) {
        return makeError(Error::InvalidArgument);
    }

    return resultFromVk(impl_->device->functions.vkGetQueryPoolResults(
        impl_->device->device,
        impl_->queryPool,
        firstQuery,
        static_cast<uint32_t>(outCompactedSizes.size()),
        static_cast<size_t>(outCompactedSizes.size()) * sizeof(uint64_t),
        outCompactedSizes.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT));
}

METALLIC_RHI_HANDLE_DEFINITIONS(Semaphore)

Result<> Semaphore::wait(uint64_t value, uint64_t timeoutNanoseconds)
{
    if (impl_ == nullptr || impl_->semaphore == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }

    const RHIOperationScope waitMarker(RHIOperation::TimelineWait, value);

    VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &impl_->semaphore,
        .pValues = &value,
    };
    const VkResult result = impl_->device->functions.vkWaitSemaphores(impl_->device->device, &waitInfo, timeoutNanoseconds);
    if (result == VK_TIMEOUT) {
        return makeError(Error::Failure);
    }
    return resultFromVk(result);
}

Result<> Semaphore::signal(uint64_t value)
{
    if (impl_ == nullptr || impl_->semaphore == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }

    VkSemaphoreSignalInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = impl_->semaphore,
        .value = value,
    };
    return resultFromVk(impl_->device->functions.vkSignalSemaphore(impl_->device->device, &signalInfo));
}

uint64_t Semaphore::currentValue() const
{
    if (impl_ == nullptr || impl_->semaphore == VK_NULL_HANDLE) {
        return 0;
    }

    uint64_t value = 0;
    const VkResult result = impl_->device->functions.vkGetSemaphoreCounterValue(impl_->device->device, impl_->semaphore, &value);
    if (result != VK_SUCCESS) {
        return 0;
    }
    return value;
}

METALLIC_RHI_HANDLE_DEFINITIONS(SwapchainSemaphore)

METALLIC_RHI_HANDLE_DEFINITIONS(Buffer)

std::shared_ptr<void> Buffer::retainAllocation() const
{
    return impl_;
}

const void* Buffer::deviceIdentity() const
{
    return impl_ ? impl_->device : nullptr;
}

std::shared_ptr<void> RayTracingAccelerationStructure::retainAllocation() const
{
    return impl_;
}

const void* RayTracingAccelerationStructure::deviceIdentity() const
{
    return impl_ ? impl_->device : nullptr;
}

const BufferDesc& Buffer::desc() const
{
    static const BufferDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

ResourceMemoryInfo Buffer::memoryInfo() const
{
    return impl_ ? impl_->memoryInfo : ResourceMemoryInfo{};
}

uint64_t Buffer::deviceAddress() const
{
    return impl_ ? impl_->address : 0;
}

Result<BufferSlice> Buffer::slice(BufferRange range) const
{
    BufferSlice whole;
    whole.allocation_ = impl_;
    whole.size_ = desc().size;
    return whole.subslice(range);
}

const BufferDesc& BufferSlice::allocationDesc() const
{
    static const BufferDesc empty;
    return allocation_ ? allocation_->desc : empty;
}

ResourceMemoryInfo BufferSlice::memoryInfo() const
{
    return allocation_ ? allocation_->memoryInfo : ResourceMemoryInfo{};
}

const void* BufferSlice::deviceIdentity() const
{
    return allocation_ ? allocation_->device : nullptr;
}

uint64_t BufferSlice::deviceAddress() const
{
    return allocation_ && allocation_->address && offset_ <= UINT64_MAX - allocation_->address
        ? allocation_->address + offset_ : 0;
}

std::shared_ptr<void> BufferSlice::retainAllocation() const
{
    return allocation_;
}

Result<BufferSlice> BufferSlice::subslice(BufferRange range) const
{
    BufferSlice next;
    const auto resolved = range.resolve(size_);
    if (!allocation_ || !resolved) {
        return makeError(Error::InvalidArgument);
    }
    next.allocation_ = allocation_;
    next.offset_ = offset_ + resolved->offset;
    next.size_ = resolved->size;
    return next;
}

Result<> BufferSlice::validate(
    const void* device,
    BufferUsageBits usage,
    uint64_t alignment,
    uint64_t minimumSize) const
{
    const auto address = deviceAddress();
    if (!allocation_ || !device || deviceIdentity() != device || size_ == 0 || size_ < minimumSize ||
        alignment == 0 || (alignment & (alignment - 1)) || !address || (address & (alignment - 1)) ||
        size_ - 1 > UINT64_MAX - address ||
        (uint32_t(allocationDesc().usage) & uint32_t(usage)) != uint32_t(usage)) {
        return makeError(Error::InvalidArgument);
    }
    return {};
}

METALLIC_RHI_HANDLE_DEFINITIONS(RayTracingAccelerationStructure)

METALLIC_RHI_HANDLE_DEFINITIONS(RayTracingBottomLevelBuildPlan)

bool RayTracingBottomLevelBuildPlan::valid() const
{
    return impl_ && impl_->device && !impl_->geometries.empty() && impl_->sizes.accelerationStructureSize != 0;
}

const RayTracingAccelerationStructureBuildSizes& RayTracingBottomLevelBuildPlan::sizes() const
{
    static const RayTracingAccelerationStructureBuildSizes empty;
    return impl_ ? impl_->sizes : empty;
}

RayTracingCoverageAccelerationStats RayTracingBottomLevelBuildPlan::coverageStats() const
{
    return impl_ ? impl_->coverageStats : RayTracingCoverageAccelerationStats{};
}

const RayTracingAccelerationStructureDesc& RayTracingAccelerationStructure::desc() const
{
    static const RayTracingAccelerationStructureDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

ResourceMemoryInfo RayTracingAccelerationStructure::memoryInfo() const
{
    return impl_ && impl_->storage ? impl_->storage->memoryInfo() : ResourceMemoryInfo{};
}

RayTracingCoverageAccelerationStats RayTracingAccelerationStructure::coverageStats() const
{
    return impl_ ? impl_->coverageStats : RayTracingCoverageAccelerationStats{};
}

bool RayTracingAccelerationStructure::valid() const
{
    if (impl_ && impl_->desc.topLevelBackend == RayTracingTopLevelBackend::Partitioned) {
        return impl_->desc.type == RayTracingAccelerationStructureType::TopLevel &&
            impl_->partitioned && impl_->storage && impl_->address != 0;
    }
    return impl_ != nullptr &&
        (impl_->micromap != VK_NULL_HANDLE ||
            (impl_->accelerationStructure != VK_NULL_HANDLE && impl_->address != 0));
}

uint64_t RayTracingAccelerationStructure::deviceAddress() const
{
    return valid() ? impl_->address : 0;
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

uint64_t Buffer::hostWriteAlignment() const
{
    if (!impl_ || !impl_->allocation) { return 1; }
    VkMemoryPropertyFlags flags = 0;
    vmaGetAllocationMemoryProperties(impl_->device->allocator, impl_->allocation, &flags);
    if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) { return 1; }
    const VkPhysicalDeviceProperties* properties = nullptr;
    vmaGetPhysicalDeviceProperties(impl_->device->allocator, &properties);
    return properties->limits.nonCoherentAtomSize;
}

void Buffer::flush(BufferRange range)
{
    if (impl_ == nullptr || impl_->allocation == VK_NULL_HANDLE) {
        return;
    }

    const VkDeviceSize vkSize = range.size == UINT64_MAX ? VK_WHOLE_SIZE : range.size;
    vmaFlushAllocation(impl_->device->allocator, impl_->allocation, range.offset, vkSize);
}

void Buffer::invalidate(BufferRange range)
{
    if (impl_ == nullptr || impl_->allocation == VK_NULL_HANDLE) {
        return;
    }

    const VkDeviceSize vkSize = range.size == UINT64_MAX ? VK_WHOLE_SIZE : range.size;
    vmaInvalidateAllocation(impl_->device->allocator, impl_->allocation, range.offset, vkSize);
}

METALLIC_RHI_HANDLE_DEFINITIONS(BufferView)

const BufferViewDesc& BufferView::desc() const
{
    static const BufferViewDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

BufferSlice BufferView::slice() const
{
    BufferSlice slice;
    if (impl_) {
        slice.allocation_ = impl_->buffer;
        slice.offset_ = impl_->desc.range.offset;
        slice.size_ = impl_->size;
    }
    return slice;
}

METALLIC_RHI_HANDLE_DEFINITIONS(Texture)

const TextureDesc& Texture::desc() const
{
    static const TextureDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

uint64_t Texture::allocationSize() const
{
    return impl_ ? impl_->allocationSize : 0;
}

ResourceMemoryInfo Texture::memoryInfo() const
{
    return impl_ ? impl_->memoryInfo : ResourceMemoryInfo{};
}

std::shared_ptr<void> Texture::retainAllocation() const
{
    return impl_ && impl_->ownsImage ? impl_ : nullptr;
}

const void* Texture::deviceIdentity() const
{
    return impl_ ? impl_->device : nullptr;
}

METALLIC_RHI_HANDLE_DEFINITIONS(TextureView)

const TextureViewDesc& TextureView::desc() const
{
    static const TextureViewDesc empty;
    return impl_ ? impl_->desc : empty;
}

std::shared_ptr<void> TextureView::retainTexture() const
{
    return impl_ && impl_->texture && impl_->texture->ownsImage ? impl_->texture : nullptr;
}

const void* TextureView::deviceIdentity() const
{
    return impl_ ? impl_->device : nullptr;
}

const void* CommandBuffer::deviceIdentity() const
{
    return impl_ ? impl_->device : nullptr;
}

METALLIC_RHI_HANDLE_DEFINITIONS(ShaderModule)

uint64_t ShaderModule::contentHash() const
{
    return impl_ != nullptr ? impl_->contentHash : 0;
}

uint64_t ShaderModule::inputSpirvHash() const
{
    return impl_ != nullptr ? impl_->inputSpirvFnv1a64 : 0;
}

METALLIC_RHI_HANDLE_DEFINITIONS(PipelineCache)

PipelineCacheStats PipelineCache::stats() const
{
    if (impl_ == nullptr) {
        return {};
    }
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}

Result<> PipelineCache::save()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return impl_->save();
}

const void* PreparedExecution::deviceIdentity() const
{
    return compute_ ? compute_->device : graphics_ ? graphics_->device : shaders_ ? shaders_->device : nullptr;
}

PreparedExecution ComputePipeline::execution() const
{
    PreparedExecution result; result.compute_ = impl_; return result;
}

PreparedExecution GraphicsPipeline::execution() const
{
    PreparedExecution result; result.graphics_ = impl_; return result;
}

PreparedExecution GraphicsShaderObjectProgram::execution(const RasterExecutionState& state) const
{
    PreparedExecution result; result.shaders_ = impl_; result.raster_ = state; return result;
}

ShaderObjectCacheStats GraphicsShaderObjectProgram::cacheStats() const
{
    return impl_ ? impl_->cacheStats : ShaderObjectCacheStats{};
}

const char* GraphicsShaderObjectProgram::binaryCacheFilePath() const
{
    return impl_ ? impl_->binaryCacheFilePath.c_str() : "";
}

METALLIC_RHI_HANDLE_DEFINITIONS(GraphicsPipeline)

namespace detail {
GraphicsPipelineImpl::~GraphicsPipelineImpl()
{
    if (device != nullptr) {
        if (pipeline != VK_NULL_HANDLE) {
            device->functions.vkDestroyPipeline(device->device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        if (layout != VK_NULL_HANDLE) {
            device->functions.vkDestroyPipelineLayout(device->device, layout, nullptr);
            layout = VK_NULL_HANDLE;
        }
    }
}
} // namespace detail

uint64_t GraphicsPipeline::psoHash() const
{
    return impl_ != nullptr ? impl_->psoHash : 0;
}

bool GraphicsPipeline::pipelineCacheHit() const
{
    return impl_ != nullptr && impl_->pipelineCacheHit;
}

METALLIC_RHI_HANDLE_DEFINITIONS(ComputePipeline)

namespace detail {
ComputePipelineImpl::~ComputePipelineImpl()
{
    if (device != nullptr) {
        if (pipeline != VK_NULL_HANDLE) {
            device->functions.vkDestroyPipeline(device->device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        if (layout != VK_NULL_HANDLE) {
            device->functions.vkDestroyPipelineLayout(device->device, layout, nullptr);
            layout = VK_NULL_HANDLE;
        }
    }
}
} // namespace detail

uint64_t ComputePipeline::psoHash() const
{
    return impl_ != nullptr ? impl_->psoHash : 0;
}

bool ComputePipeline::pipelineCacheHit() const
{
    return impl_ != nullptr && impl_->pipelineCacheHit;
}

METALLIC_RHI_HANDLE_DEFINITIONS(GraphicsShaderObjectProgram)

namespace detail {
GraphicsShaderObjectProgramImpl::~GraphicsShaderObjectProgramImpl()
{
    if (device != nullptr) {
        if (vertexShader != VK_NULL_HANDLE) {
            device->functions.vkDestroyShaderEXT(device->device, vertexShader, nullptr);
            vertexShader = VK_NULL_HANDLE;
        }
        if (fragmentShader != VK_NULL_HANDLE) {
            device->functions.vkDestroyShaderEXT(device->device, fragmentShader, nullptr);
            fragmentShader = VK_NULL_HANDLE;
        }
    }
}
} // namespace detail

METALLIC_RHI_HANDLE_DEFINITIONS(BindlessHeap)

const BindlessHeapDesc& BindlessHeap::desc() const
{
    static const BindlessHeapDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}


Result<BindlessHandle> BindlessHeap::allocate(BindlessHandleKind kind)
{
    if (impl_ == nullptr || kind < BindlessHandleKind::Sampler || kind > BindlessHandleKind::AccelerationStructure) {
        return makeError(Error::InvalidArgument);
    }
    BindlessHandle handle{};
    if (!impl_->heap.allocate(kind, handle)) { return makeError(Error::OutOfMemory); }
    return handle;
}

void BindlessHeap::release(BindlessHandle handle)
{
    if (impl_ != nullptr) {
        impl_->heap.release(handle);
    }
}

Result<> BindlessHeap::writeSampler(BindlessHandle handle, const SamplerDesc& sampler)
{
    const BindlessSamplerWrite write{
        .handle = handle,
        .sampler = sampler,
    };
    return writeSamplers({&write, 1});
}

Result<> BindlessHeap::writeSamplers(std::span<const BindlessSamplerWrite> writes)
{
    if (impl_ == nullptr ||
        impl_->samplerHeap.mapped == nullptr ||
        writes.size() > UINT32_MAX ||
        writes.size() == 0) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<BindlessHandle> handles(writes.size());
    std::vector<VkSamplerCreateInfo> samplerInfos(writes.size());
    for (uint32_t index = 0; index < writes.size(); ++index) {
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
        writes.size(),
        impl_->samplerHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushSamplerDirty();
    return {};
}

Result<> BindlessHeap::writeSampledImage(BindlessHandle handle, TextureView& view, TextureLayout layout)
{
    const BindlessImageWrite write{
        .handle = handle,
        .view = &view,
        .layout = layout,
    };
    return writeImages({&write, 1});
}

Result<> BindlessHeap::writeStorageImage(BindlessHandle handle, TextureView& view)
{
    const BindlessImageWrite write{
        .handle = handle,
        .view = &view,
        .layout = TextureLayout::General,
    };
    return writeImages({&write, 1});
}

Result<> BindlessHeap::writeImages(std::span<const BindlessImageWrite> writes)
{
    if (impl_ == nullptr ||
        impl_->resourceHeap.mapped == nullptr ||
        writes.size() > UINT32_MAX ||
        writes.size() == 0) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<BindlessHandle> handles(writes.size());
    std::vector<VkImageViewCreateInfo> viewInfos(writes.size());
    std::vector<VkImageDescriptorInfoEXT> imageInfos(writes.size());
    std::vector<VkResourceDescriptorInfoEXT> resourceInfos(writes.size());
    for (uint32_t index = 0; index < writes.size(); ++index) {
        const BindlessImageWrite& write = writes[index];
        TextureView* view = write.view;
        const bool sampled = write.handle.kind == BindlessHandleKind::SampledImage;
        const bool storage = write.handle.kind == BindlessHandleKind::StorageImage;
        if ((!sampled && !storage) ||
            view == nullptr ||
            view->impl_ == nullptr ||
            view->impl_->texture == nullptr || view->impl_->device != impl_->device) {
            return makeError(Error::InvalidArgument);
        }

        const TextureDesc& textureDesc = view->impl_->texture->desc;
        if ((sampled && !hasFlag(textureDesc.usage, TextureUsageBits::Sampled)) ||
            (storage && !hasFlag(textureDesc.usage, TextureUsageBits::Storage))) {
            return makeError(Error::InvalidArgument);
        }
        handles[index] = write.handle;
        viewInfos[index] = view->impl_->createInfo();
        imageInfos[index] = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
            .pView = &viewInfos[index],
            .layout = imageLayout(write.layout, impl_->device->vulkanCapabilities.unifiedImageLayouts),
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
        writes.size(),
        impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

Result<> BindlessHeap::writeBufferView(BindlessHandle handle, BufferView& view)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr || view.impl_ == nullptr ||
        view.impl_->device != impl_->device || !view.impl_->buffer ||
        view.impl_->buffer->device != impl_->device ||
        !hasFlag(view.impl_->buffer->desc.usage, view.impl_->descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            ? BufferUsageBits::Constant : BufferUsageBits::Storage)) {
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

Result<> BindlessHeap::writeConstantBuffer(BindlessHandle handle, Buffer& buffer)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr || buffer.impl_ == nullptr ||
        buffer.impl_->device != impl_->device || !hasFlag(buffer.impl_->desc.usage, BufferUsageBits::Constant)) {
        return makeError(Error::InvalidArgument);
    }

    const VkDeviceAddress address = buffer.impl_->address;
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

Result<> BindlessHeap::writeStorageBuffer(BindlessHandle handle, const BufferSlice& buffer)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr || !buffer.allocation_ ||
        buffer.allocation_->device != impl_->device ||
        (uint32_t(buffer.allocationDesc().usage) & uint32_t(BufferUsageBits::Storage)) == 0) {
        return makeError(Error::InvalidArgument);
    }

    const VkDeviceAddress address = buffer.allocation_->address;
    const VkResult result = impl_->heap.writeBufferDescriptor(
        handle,
        address,
        buffer.allocationDesc().size,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

Result<> BindlessHeap::writeAccelerationStructure(
    BindlessHandle handle,
    RayTracingAccelerationStructure& accelerationStructure)
{
    if (impl_ == nullptr || impl_->resourceHeap.mapped == nullptr ||
        accelerationStructure.impl_ == nullptr || !accelerationStructure.valid() ||
        accelerationStructure.impl_->device != impl_->device ||
        accelerationStructure.desc().type != RayTracingAccelerationStructureType::TopLevel) {
        return makeError(Error::InvalidArgument);
    }

    const VkDeviceAddress address = accelerationStructure.deviceAddress();
    const VkResult result = accelerationStructure.desc().topLevelBackend == RayTracingTopLevelBackend::Partitioned
        ? impl_->heap.writePartitionedAccelerationStructureDescriptor(handle, address, 0, impl_->resourceHeap.mapped)
        : impl_->heap.writeAccelerationStructureDescriptor(handle, address, 0, impl_->resourceHeap.mapped);
    if (result != VK_SUCCESS) {
        return resultFromVk(result);
    }
    impl_->flushResourceDirty();
    return {};
}

METALLIC_RHI_HANDLE_CONSTRUCTORS(CommandBuffer)

CommandBuffer::~CommandBuffer()
{
    if (submission_) { submission_->owner = nullptr; submission_->cancel(); }
    if (impl_ != nullptr && impl_->commandBuffer != VK_NULL_HANDLE) {
        vulkan::forgetTraceObject(impl_->device->device, VK_OBJECT_TYPE_COMMAND_BUFFER, uint64_t(impl_->commandBuffer));
        if (impl_->capturePool != nullptr) {
            // Nsight 2026.3.1 retains freed wrappers in its event polling list.
            // Reset alone does not remove those references. Keep native handles
            // alive and reuse them until device teardown, including across pool
            // lifetimes. Callers must still complete work before destruction.
            // TODO: Remove after an updated Nsight passes the full-size capture
            // and resize regression without reuse; see the capture investigation.
            (void)impl_->device->functions.vkResetCommandBuffer(impl_->commandBuffer, 0);
            impl_->capturePool->availableBuffers.push_back(impl_->commandBuffer);
        } else {
            impl_->device->functions.vkFreeCommandBuffers(impl_->device->device, impl_->pool, 1, &impl_->commandBuffer);
        }
        impl_->commandBuffer = VK_NULL_HANDLE;
    }
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : impl_(std::move(other.impl_)), submissionContext_(std::move(other.submissionContext_)),
      submission_(std::move(other.submission_)),
      dependencyWaits_(std::move(other.dependencyWaits_)), dependencyLifetimes_(std::move(other.dependencyLifetimes_)),
      recording_(std::exchange(other.recording_, false))
{
    if (submission_) { submission_->owner = this; }
}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept
{
    if (this != &other) {
        this->~CommandBuffer();
        new (this) CommandBuffer(std::move(other));
    }
    return *this;
}

QueueAccessBits CommandBuffer::queueCapabilities() const
{
    if (!impl_) { return QueueAccessBits::None; }
    QueueAccessBits result = QueueAccessBits::None;
    if (impl_->queueFlags & VK_QUEUE_GRAPHICS_BIT) { result = result | QueueAccessBits::Graphics; }
    if (impl_->queueFlags & VK_QUEUE_COMPUTE_BIT) { result = result | QueueAccessBits::Compute; }
    if (impl_->queueFlags & VK_QUEUE_TRANSFER_BIT) { result = result | QueueAccessBits::Copy; }
    return result;
}

namespace {

// Recording state and queue compatibility are argument errors, independent of optional device features.
bool validCommandRecording(const detail::CommandBufferImpl* commandBuffer, bool recording,
    VkQueueFlags supportedQueues = 0)
{
    return commandBuffer && recording &&
        (!supportedQueues || (commandBuffer->queueFlags & supportedQueues));
}

} // namespace

Result<> CommandBuffer::begin(std::shared_ptr<CommandSubmissionContext> context)
{
    if (impl_ == nullptr || (context && !context->recording()) ||
        (submission_ && submission_->sealed && !submission_->submitted && !submission_->cancelled)) {
        return makeError(Error::InvalidArgument);
    }

    impl_->currentGraphicsPipelineLayout = VK_NULL_HANDLE;
    impl_->currentComputePipelineLayout = VK_NULL_HANDLE;
    impl_->currentComputePipeline = VK_NULL_HANDLE;
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
    impl_->synchronizationStats = {};
    Result<> result = resultFromVk(impl_->device->functions.vkBeginCommandBuffer(impl_->commandBuffer, &beginInfo));
    recording_ = result.has_value();
    if (result) {
        vulkan::emitTrace({.kind = vulkan::TraceKind::CommandBegin, .device = impl_->device->device,
            .command = impl_->commandBuffer});
        if (submission_ != nullptr) { submission_->cancel(); }
        submission_ = std::make_shared<detail::CommandSubmissionState>();
        submission_->owner = this;
        impl_->submissions->add(submission_);
        if (context) { context->registerRecording(submission_); }
        dependencyWaits_.clear();
        dependencyLifetimes_.clear();
    }
    submissionContext_ = result ? std::move(context) : nullptr;
    return result;
}

Result<> CommandBuffer::end()
{
    if (!validCommandRecording(impl_.get(), recording_) || !submission_) {
        return makeError(Error::InvalidArgument);
    }
    Result<> result = resultFromVk(impl_->device->functions.vkEndCommandBuffer(impl_->commandBuffer));
    if (result) {
        recording_ = false;
        submission_->finished.store(true, std::memory_order_release);
    }
    return result;
}

void CommandBuffer::beginDebugLabel(const DebugLabelDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_) ||
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
    if (!validCommandRecording(impl_.get(), recording_) ||
        impl_->device == nullptr ||
        impl_->device->cmdEndDebugUtilsLabel == nullptr) {
        return;
    }

    impl_->device->cmdEndDebugUtilsLabel(impl_->commandBuffer);
}

Result<> CommandBuffer::resetTimestampQueries(
    TimestampQueryPool& queryPool,
    uint32_t firstQuery,
    uint32_t queryCount)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) ||
        queryPool.impl_ == nullptr ||
        impl_->device != queryPool.impl_->device ||
        queryCount == 0 ||
        firstQuery >= queryPool.impl_->desc.queryCount ||
        queryCount > queryPool.impl_->desc.queryCount - firstQuery) {
        return makeError(Error::InvalidArgument);
    }

    impl_->device->functions.vkCmdResetQueryPool(
        impl_->commandBuffer,
        queryPool.impl_->queryPool,
        firstQuery,
        queryCount);
    return {};
}

Result<> CommandBuffer::writeTimestamp(
    TimestampQueryPool& queryPool,
    uint32_t queryIndex,
    PipelineStageBits stage)
{
    if (!validCommandRecording(impl_.get(), recording_) ||
        queryPool.impl_ == nullptr ||
        impl_->device != queryPool.impl_->device ||
        queryPool.impl_->queueFamilyIndex != impl_->queueFamilyIndex ||
        queryIndex >= queryPool.impl_->desc.queryCount) {
        return makeError(Error::InvalidArgument);
    }

    if (!vulkan::validDeviceStages(stage, syncSupport(*impl_->device, impl_->queueFlags))) {
        return makeError(Error::InvalidArgument);
    }
    const auto nativeStage = toVkPipelineStages(stage);
    if (!nativeStage || (nativeStage & (nativeStage - 1)) != 0) { return makeError(Error::InvalidArgument); }
    impl_->device->functions.vkCmdWriteTimestamp2(
        impl_->commandBuffer,
        nativeStage,
        queryPool.impl_->queryPool,
        queryIndex);
    return {};
}

Result<> CommandBuffer::resetRayTracingAccelerationStructureCompactionQueries(
    RayTracingAccelerationStructureCompactionQueryPool& queryPool,
    uint32_t firstQuery,
    uint32_t queryCount)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT) ||
        queryPool.impl_ == nullptr ||
        impl_->device != queryPool.impl_->device ||
        queryCount == 0 ||
        firstQuery >= queryPool.impl_->desc.queryCount ||
        queryCount > queryPool.impl_->desc.queryCount - firstQuery) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->device->capabilities.rayTracingAccelerationStructure ||
        !impl_->device->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    impl_->device->functions.vkCmdResetQueryPool(
        impl_->commandBuffer,
        queryPool.impl_->queryPool,
        firstQuery,
        queryCount);
    return {};
}

Result<> CommandBuffer::writeRayTracingAccelerationStructureCompactedSize(
    RayTracingAccelerationStructureCompactionQueryPool& queryPool,
    uint32_t queryIndex,
    RayTracingAccelerationStructure& accelerationStructure)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT) ||
        queryPool.impl_ == nullptr ||
        impl_->device != queryPool.impl_->device ||
        queryIndex >= queryPool.impl_->desc.queryCount ||
        accelerationStructure.impl_ == nullptr ||
        accelerationStructure.impl_->device != impl_->device ||
        !accelerationStructure.valid() ||
        accelerationStructure.impl_->accelerationStructure == VK_NULL_HANDLE ||
        !hasFlag(
            accelerationStructure.impl_->desc.buildFlags,
            RayTracingAccelerationStructureBuildFlags::AllowCompaction)) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->device->capabilities.rayTracingAccelerationStructure ||
        !impl_->device->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    const auto retained = retainResource(accelerationStructure.retainAllocation());
    if (!retained) { return retained; }

    const VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, dependency);

    const VkAccelerationStructureKHR nativeAccelerationStructure =
        accelerationStructure.impl_->accelerationStructure;
    impl_->device->functions.vkCmdWriteAccelerationStructuresPropertiesKHR(
        impl_->commandBuffer,
        1,
        &nativeAccelerationStructure,
        VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
        queryPool.impl_->queryPool,
        queryIndex);
    return {};
}

SynchronizationStats CommandBuffer::synchronizationStats() const
{
    return impl_ ? impl_->synchronizationStats : SynchronizationStats{};
}

Result<> CommandBuffer::synchronize(const BarrierDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_) || (desc.textures.size() > UINT32_MAX) ||
        (desc.buffers.size() > UINT32_MAX) || (desc.memory.size() > UINT32_MAX) ||
        (desc.accelerationStructures.size() > UINT32_MAX)) {
        return makeError(Error::InvalidArgument);
    }
    const auto support = syncSupport(*impl_->device, impl_->queueFlags);
    const auto validScope = [&](SyncScope scope) { return vulkan::validScope(scope, support); };
    std::vector<VkImageMemoryBarrier2> images;
    std::vector<VkMemoryBarrier2> memory;
    uint64_t coalesced = 0;
    const auto appendMemory = [&](VulkanSyncScope before, VulkanSyncScope after) {
        if (!before.stage && !after.stage) { return; }
        // Preserve stage pairs; unioning unrelated pairs would add false ordering.
        for (auto& existing : memory) {
            if (existing.srcStageMask == before.stage && existing.dstStageMask == after.stage) {
                existing.srcAccessMask |= before.access;
                existing.dstAccessMask |= after.access;
                return;
            }
        }
        memory.push_back({.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = before.stage, .srcAccessMask = before.access,
            .dstStageMask = after.stage, .dstAccessMask = after.access});
    };
    const auto resourceMemory = [&](VulkanSyncScope before, VulkanSyncScope after) {
        // Explicit scopes are authoritative, including read/read and execution-only
        // dependencies. Hazard elision belongs to the access planner.
        if (!before.stage && !after.stage) { return; }
        appendMemory(before, after);
        ++coalesced;
    };
    for (uint32_t i = 0; i < desc.memory.size(); ++i) {
        const auto& barrier = desc.memory[i];
        if (!validScope(barrier.before) || !validScope(barrier.after)) { return makeError(Error::InvalidArgument); }
        appendMemory(scopeInfo(barrier.before), scopeInfo(barrier.after));
    }
    for (uint32_t i = 0; i < desc.textures.size(); ++i) {
        const auto& barrier = desc.textures[i];
        if (!barrier.texture || !barrier.texture->impl_ || barrier.texture->impl_->device != impl_->device ||
            !validScope(barrier.before) || !validScope(barrier.after)) { return makeError(Error::InvalidArgument); }
        const auto& texture = *barrier.texture->impl_;
        if (!barrier.range.valid(texture.desc.mipCount, texture.desc.layerCount)) { return makeError(Error::InvalidArgument); }
        const auto before = scopeInfo(barrier.before);
        const auto after = scopeInfo(barrier.after);
        if (barrier.oldLayout > TextureLayout::General || barrier.newLayout > TextureLayout::General ||
            barrier.newLayout == TextureLayout::Undefined) { return makeError(Error::InvalidArgument); }
        const auto oldLayout = imageLayout(barrier.oldLayout, impl_->device->vulkanCapabilities.unifiedImageLayouts);
        const auto newLayout = imageLayout(barrier.newLayout, impl_->device->vulkanCapabilities.unifiedImageLayouts);
        if (newLayout == VK_IMAGE_LAYOUT_UNDEFINED) { return makeError(Error::InvalidArgument); }
        if (oldLayout == newLayout) { resourceMemory(before, after); continue; }
        images.push_back({.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = before.stage, .srcAccessMask = before.access, .dstStageMask = after.stage, .dstAccessMask = after.access,
            .oldLayout = oldLayout, .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = texture.image,
            .subresourceRange = {aspectForFormat(texture.desc.format), barrier.range.baseMip, barrier.range.mipCount, barrier.range.baseLayer, barrier.range.layerCount}});
    }
    for (uint32_t i = 0; i < desc.buffers.size(); ++i) {
        const auto& barrier = desc.buffers[i];
        if (!barrier.buffer || barrier.buffer->deviceIdentity() != deviceIdentity() ||
            !validScope(barrier.before) || !validScope(barrier.after)) { return makeError(Error::InvalidArgument); }
        const auto range = barrier.range.resolve(barrier.buffer->desc().size);
        if (!range || !range->size) { return makeError(Error::InvalidArgument); }
        const auto before = scopeInfo(barrier.before);
        const auto after = scopeInfo(barrier.after);
        resourceMemory(before, after);
    }
    for (const auto& barrier : desc.accelerationStructures) {
        auto* accelerationStructure = barrier.accelerationStructure;
        if (!accelerationStructure || !accelerationStructure->valid() ||
            accelerationStructure->deviceIdentity() != deviceIdentity() ||
            !validScope(barrier.before) || !validScope(barrier.after)) {
            return makeError(Error::InvalidArgument);
        }
        const auto families = detail::queueFamiliesForAccess(*impl_->device,
            accelerationStructure->impl_->storage->desc().queueAccess);
        if (std::find(families.begin(), families.end(), impl_->queueFamilyIndex) == families.end()) {
            return makeError(Error::InvalidArgument);
        }
        resourceMemory(scopeInfo(barrier.before), scopeInfo(barrier.after));
    }
    if (images.empty() && memory.empty()) { return {}; }
    // Everything was validated before retaining resources or recording Vulkan commands.
    for (uint32_t i = 0; i < desc.textures.size(); ++i) {
        auto result = retainResource(desc.textures[i].texture->impl_);
        if (!result) { return result; }
    }
    for (const auto& barrier : desc.accelerationStructures) {
        auto result = retainResource(barrier.accelerationStructure->retainAllocation());
        if (!result) { return result; }
    }
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = uint32_t(memory.size()), .pMemoryBarriers = memory.data(),
        .imageMemoryBarrierCount = uint32_t(images.size()), .pImageMemoryBarriers = images.data()};
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, dependency, &desc);
    auto& stats = impl_->synchronizationStats;
    ++stats.calls; stats.memoryBarriers += memory.size(); stats.imageTransitions += images.size(); stats.coalescedResources += coalesced;
    return {};
}

Result<> CommandBuffer::copyBuffer(const BufferSlice& source, const BufferSlice& destination)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) ||
        !source.validate(deviceIdentity(), BufferUsageBits::TransferSource) ||
        !destination.validate(deviceIdentity(), BufferUsageBits::TransferDestination) || source.size() != destination.size()) {
        return makeError(Error::InvalidArgument);
    }
    if (detail::BufferAddressCommandAccess::overlap(source, destination)) { return makeError(Error::InvalidArgument); }
    auto result = retainResource(source.retainAllocation());
    if (result) { result = retainResource(destination.retainAllocation()); }
    if (!result) { return result; }
    const VkDeviceMemoryCopyKHR copyRegion{
        .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_COPY_KHR,
        .srcRange = {source.deviceAddress(), source.size()}, .srcFlags = detail::BufferAddressCommandAccess::flags(source),
        .dstRange = {destination.deviceAddress(), destination.size()}, .dstFlags = detail::BufferAddressCommandAccess::flags(destination),
    };
    const VkCopyDeviceMemoryInfoKHR copyInfo{.sType = VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_INFO_KHR,
        .regionCount = 1, .pRegions = &copyRegion};
    impl_->device->functions.vkCmdCopyMemoryKHR(impl_->commandBuffer, &copyInfo);
    return {};
}

namespace {

bool queueCanAccessBuffer(const detail::CommandBufferImpl& commands, const BufferDesc& buffer)
{
    const auto families = detail::queueFamiliesForAccess(*commands.device, buffer.queueAccess);
    return std::find(families.begin(), families.end(), commands.queueFamilyIndex) != families.end();
}

bool validCommandSlice(const detail::CommandBufferImpl& commands, const BufferSlice& slice,
    BufferUsageBits usage, uint64_t minimumBytes = 1, uint64_t alignment = 1)
{
    return slice.validate(commands.device, usage, alignment, minimumBytes) &&
        queueCanAccessBuffer(commands, slice.allocationDesc());
}

bool validTriangleGeometry(const detail::DeviceImpl* device, const RayTracingTriangleGeometryDesc& source)
{
    const auto info = formatInfo(source.vertexFormat);
    if (!source.vertexCount || !source.vertexStride || !source.primitiveCount ||
        info.blockExtent != 1 || !info.bytesPerBlock || source.vertexStride < info.bytesPerBlock ||
        uint64_t(source.vertexCount - 1) > (UINT64_MAX - info.bytesPerBlock) / source.vertexStride ||
        !source.vertexBuffer.validate(device, BufferUsageBits::AccelerationStructureBuildInput, 1,
            uint64_t(source.vertexCount - 1) * source.vertexStride + info.bytesPerBlock)) {
        return false;
    }
    if (source.indexType == RayTracingIndexType::None) { return uint64_t(source.primitiveCount) * 3 <= source.vertexCount; }
    const uint64_t indexBytes = source.indexType == RayTracingIndexType::Uint16 ? 2 :
        source.indexType == RayTracingIndexType::Uint32 ? 4 : 0;
    return indexBytes && source.indexBuffer.validate(device, BufferUsageBits::AccelerationStructureBuildInput,
        indexBytes, uint64_t(source.primitiveCount) * 3 * indexBytes).has_value();
}

uint64_t rayTracingScratchAlignment(const detail::DeviceImpl& device)
{
    return std::max<uint64_t>({device.physicalProperties.accelerationStructure.minAccelerationStructureScratchOffsetAlignment,
        device.capabilities.partitionedAccelerationStructure ? 256ull : 1ull,
        device.vulkanCapabilities.opacityMicromap ? (device.opacityMicromapExt ? 128ull : 256ull) : 1ull});
}

Result<BufferSlice> alignedBuildScratch(const detail::CommandBufferImpl& commands,
    const BufferSlice& scratch, uint64_t alignment, uint64_t minimumBytes)
{
    if (!validCommandSlice(commands, scratch, BufferUsageBits::Storage) ||
        !alignment || (alignment & (alignment - 1))) { return makeError(Error::InvalidArgument); }
    const uint64_t padding = (0 - scratch.deviceAddress()) & (alignment - 1);
    auto aligned = scratch.subslice({padding});
    if (!aligned || !aligned->validate(commands.device, BufferUsageBits::Storage, alignment, minimumBytes)) {
        return makeError(Error::InvalidArgument);
    }
    return aligned;
}

Result<RayTracingAccelerationStructureBuildSizes> queryBottomLevelBuildSizes(
    const detail::DeviceImpl& device,
    std::span<const RayTracingTriangleGeometryDesc> sources,
    std::span<const detail::PreparedGeometryCoverage> coverage,
    RayTracingAccelerationStructureBuildFlags flags)
{
    if (sources.empty() || sources.size() > UINT32_MAX ||
        (!coverage.empty() && coverage.size() != sources.size())) {
        return makeError(Error::InvalidArgument);
    }
    std::vector<VkAccelerationStructureGeometryKHR> geometries(sources.size());
    std::vector<uint32_t> primitiveCounts(sources.size());
    std::vector<VkAccelerationStructureTrianglesOpacityMicromapKHR> attachments(sources.size());
    std::vector<VkAccelerationStructureTrianglesOpacityMicromapEXT> extAttachments(sources.size());
    std::vector<std::vector<VkMicromapUsageEXT>> extUsages(sources.size());
    for (size_t index = 0; index < sources.size(); ++index) {
        const auto& source = sources[index];
        if (!validTriangleGeometry(&device, source)) { return makeError(Error::InvalidArgument); }
        const void* attachment = nullptr;
        if (!coverage.empty() && coverage[index].resource) {
            const auto& resource = *coverage[index].resource;
            if (resource.micromap != VK_NULL_HANDLE) {
                const auto attached = makeExtMicromapAttachment(source.primitiveCount,
                    coverage[index].baked.usages, resource.micromap, extUsages[index], extAttachments[index]);
                if (!attached) { return makeError(attached.error()); }
                attachment = &extAttachments[index];
            } else {
                attachments[index] = {
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_KHR,
                    .indexType = VK_INDEX_TYPE_NONE_KHR,
                    .micromap = resource.accelerationStructure,
                };
                attachment = &attachments[index];
            }
        }
        geometries[index] = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .flags = toVkGeometryFlags(source.flags),
        };
        geometries[index].geometry.triangles = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .pNext = attachment,
            .vertexFormat = toVkFormat(source.vertexFormat),
            .vertexData = {.deviceAddress = source.vertexBuffer.deviceAddress()},
            .vertexStride = source.vertexStride,
            .maxVertex = source.vertexCount - 1,
            .indexType = toVkRayTracingIndexType(source.indexType),
            .indexData = {.deviceAddress = source.indexType == RayTracingIndexType::None ? 0 : source.indexBuffer.deviceAddress()},
        };
        primitiveCounts[index] = source.primitiveCount;
    }
    const VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags = toVkAccelerationStructureBuildFlags(flags),
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = static_cast<uint32_t>(geometries.size()),
        .pGeometries = geometries.data(),
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    device.functions.vkGetAccelerationStructureBuildSizesKHR(device.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, primitiveCounts.data(), &sizes);
    if (!sizes.accelerationStructureSize || !sizes.buildScratchSize) { return makeError(Error::Failure); }
    return RayTracingAccelerationStructureBuildSizes{sizes.accelerationStructureSize, sizes.buildScratchSize, sizes.updateScratchSize};
}

Result<std::shared_ptr<detail::MicromapBufferAllocation>> createMicromapBuildUpload(
    detail::DeviceImpl& device, const detail::BakedOpacityMicromap& baked)
{
    const uint64_t triangleBytes = baked.triangles.size() * sizeof(OpacityMicromapTriangle);
    const uint64_t bytes = 127 + ((uint64_t(baked.data.size()) + 127) & ~uint64_t(127)) + triangleBytes;
    const auto families = detail::queueFamiliesForAccess(device, QueueAccessBits::Graphics | QueueAccessBits::Compute);
    const VkBufferCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VkBufferUsageFlags(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            (device.opacityMicromapExt ? VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT : 0)),
        .sharingMode = families.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = families.size() > 1 ? static_cast<uint32_t>(families.size()) : 0,
        .pQueueFamilyIndices = families.size() > 1 ? families.data() : nullptr,
    };
    auto allocationInfo = allocationInfoForMemory(MemoryLocation::HostUpload);
    auto upload = std::make_shared<detail::MicromapBufferAllocation>();
    std::unique_lock budgetLock(device.memoryBudgetState->mutex);
    const auto admitted = device.prepareBufferAllocationLocked(info, allocationInfo, MemoryBudgetDomain::Upload);
    if (!admitted) { return makeError(admitted.error()); }
    upload->device = &device;
    upload->allocator = device.allocator;
    upload->domain = MemoryBudgetDomain::Upload;
    VmaAllocationInfo allocatedInfo{};
    VkResult result = vmaCreateBuffer(device.allocator, &info, &allocationInfo, &upload->buffer, &upload->allocation, &allocatedInfo);
    if (result != VK_SUCCESS) { return makeError(resultFromVk(result).error()); }
    upload->allocationBytes = allocatedInfo.size;
    upload->deviceLocal = (device.memoryProperties.memoryHeaps[device.memoryProperties.memoryTypes[allocatedInfo.memoryType].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    device.trackMemoryLocked(upload->domain, upload->allocationBytes, upload->deviceLocal, true);
    budgetLock.unlock();
    const VkBufferDeviceAddressInfo addressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = upload->buffer};
    upload->address = device.functions.vkGetBufferDeviceAddress(device.device, &addressInfo);
    if (!upload->address) { return makeError(Error::Failure); }
    upload->dataOffset = (0 - upload->address) & 127;
    upload->triangleOffset = upload->dataOffset + ((uint64_t(baked.data.size()) + 127) & ~uint64_t(127));
    void* mapped = nullptr;
    result = vmaMapMemory(device.allocator, upload->allocation, &mapped);
    if (result != VK_SUCCESS) { return makeError(resultFromVk(result).error()); }
    std::memcpy(static_cast<uint8_t*>(mapped) + upload->dataOffset, baked.data.data(), baked.data.size());
    std::memcpy(static_cast<uint8_t*>(mapped) + upload->triangleOffset, baked.triangles.data(), triangleBytes);
    result = vmaFlushAllocation(device.allocator, upload->allocation, 0, bytes);
    vmaUnmapMemory(device.allocator, upload->allocation);
    if (result != VK_SUCCESS) { return makeError(resultFromVk(result).error()); }
    return upload;
}

struct NativeMicromapBuild {
    std::vector<VkMicromapUsageKHR> usages;
    std::vector<VkMicromapUsageEXT> extUsages;
    VkAccelerationStructureGeometryMicromapDataKHR data{};
    VkAccelerationStructureGeometryKHR geometry{};
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    VkMicromapBuildInfoEXT extBuildInfo{};
};

void prepareNativeMicromapBuild(NativeMicromapBuild& native, bool ext,
    const detail::PreparedGeometryCoverage& coverage, const detail::MicromapBufferAllocation& upload,
    VkDeviceAddress scratchAddress)
{
    native.usages.reserve(coverage.baked.usages.size());
    for (const auto& usage : coverage.baked.usages) {
        native.usages.push_back({usage.count, usage.subdivisionLevel, static_cast<VkOpacityMicromapFormatKHR>(usage.format)});
    }
    native.data = {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR,
        .usageCountsCount = static_cast<uint32_t>(native.usages.size()),
        .pUsageCounts = native.usages.data(),
        .data = upload.address + upload.dataOffset,
        .triangleArray = upload.address + upload.triangleOffset,
        .triangleArrayStride = sizeof(OpacityMicromapTriangle),
    };
    if (ext) {
        native.extBuildInfo = makeExtMicromapBuildInfo(native.data,
            RayTracingAccelerationStructureBuildFlags::PreferFastTrace, native.extUsages);
        native.extBuildInfo.dstMicromap = coverage.resource->micromap;
        native.extBuildInfo.scratchData.deviceAddress = scratchAddress;
    } else {
        native.geometry = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .pNext = &native.data, .geometryType = VK_GEOMETRY_TYPE_MICROMAP_KHR};
        native.buildInfo = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .dstAccelerationStructure = coverage.resource->accelerationStructure,
            .geometryCount = 1, .pGeometries = &native.geometry, .scratchData = {.deviceAddress = scratchAddress},
        };
    }
}

void recordPreparedMicromapBuild(detail::CommandBufferImpl& commands, const NativeMicromapBuild& native)
{
    const bool ext = commands.device->opacityMicromapExt;
    VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = ext ? VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT : VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
            (ext ? VK_ACCESS_2_MICROMAP_READ_BIT_EXT | VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT :
                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR),
    };
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
    vulkan::recordBarrier(commands.device->functions, commands.device->device, commands.commandBuffer, dependency);
    if (ext) {
        commands.device->functions.vkCmdBuildMicromapsEXT(commands.commandBuffer, 1, &native.extBuildInfo);
    } else {
        const VkAccelerationStructureBuildRangeInfoKHR* ranges = nullptr;
        commands.device->functions.vkCmdBuildAccelerationStructuresKHR(commands.commandBuffer, 1, &native.buildInfo, &ranges);
    }
    barrier.srcStageMask = barrier.dstStageMask;
    barrier.srcAccessMask = ext ? VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT : VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
        (ext ? VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT : 0);
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
        (ext ? VK_ACCESS_2_MICROMAP_READ_BIT_EXT | VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT : 0);
    vulkan::recordBarrier(commands.device->functions, commands.device->device, commands.commandBuffer, dependency);
}

Result<> retainBufferSlices(CommandBuffer& commands, std::initializer_list<BufferSlice> slices)
{
    for (const auto& slice : slices) {
        if (slice.valid()) {
            auto result = commands.retainResource(slice.retainAllocation());
            if (!result) { return result; }
        }
    }
    return {};
}

// CLAS indirect tables may occupy a subrange of an already mapped upload buffer.
Result<> uploadBufferSlice(const BufferSlice& slice, const void* data, uint64_t byteSize)
{
    auto* allocation = detail::BufferAddressCommandAccess::allocation(slice);
    if (!allocation || !allocation->allocation || !data || byteSize > slice.size()) {
        return makeError(Error::InvalidArgument);
    }
    void* mapped = nullptr;
    auto result = resultFromVk(vmaMapMemory(allocation->device->allocator, allocation->allocation, &mapped));
    if (!result) { return result; }
    std::memcpy(static_cast<uint8_t*>(mapped) + slice.offset(), data, size_t(byteSize));
    result = resultFromVk(vmaFlushAllocation(allocation->device->allocator, allocation->allocation, slice.offset(), byteSize));
    vmaUnmapMemory(allocation->device->allocator, allocation->allocation);
    return result;
}

Result<std::vector<VkDecompressMemoryRegionEXT>> prepareDecompressionRegions(
    const detail::CommandBufferImpl* commands, bool recording, std::span<const BufferDecompressionDesc> regions)
{
    if (!validCommandRecording(commands, recording, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
        return makeError(Error::InvalidArgument);
    }
    if (!commands->device->capabilities.memoryDecompression) {
        return makeError(Error::Unsupported);
    }
    if (regions.empty()) { return std::vector<VkDecompressMemoryRegionEXT>{}; }
    if (regions.size() > UINT32_MAX) { return makeError(Error::InvalidArgument); }
    std::vector<VkDecompressMemoryRegionEXT> native;
    struct Range { uint64_t begin, end; bool destination; };
    std::vector<Range> ranges;
    native.reserve(regions.size());
    ranges.reserve(regions.size() * 2);
    for (const auto& region : regions) {
        if (region.destination.size() > 65536 ||
            !validCommandSlice(*commands, region.source, BufferUsageBits::MemoryDecompression, 1, 4) ||
            !validCommandSlice(*commands, region.destination, BufferUsageBits::MemoryDecompression, 1, 4)) {
            return makeError(Error::InvalidArgument);
        }
        const uint64_t source = region.source.deviceAddress();
        const uint64_t destination = region.destination.deviceAddress();
        native.push_back({source, destination, region.source.size(), region.destination.size()});
        ranges.push_back({source, source + region.source.size(), false});
        ranges.push_back({destination, destination + region.destination.size(), true});
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) { return a.begin < b.begin; });
    uint64_t sourceEnd = 0, destinationEnd = 0;
    for (const auto& range : ranges) {
        if (range.begin < destinationEnd || (range.destination && range.begin < sourceEnd)) {
            return makeError(Error::InvalidArgument);
        }
        if (range.destination) { destinationEnd = std::max(destinationEnd, range.end); }
        else { sourceEnd = std::max(sourceEnd, range.end); }
    }
    return native;
}

} // namespace

Result<> CommandBuffer::decompressBuffers(std::span<const BufferDecompressionDesc> regions)
{
    auto native = prepareDecompressionRegions(impl_.get(), recording_, regions);
    if (!native) { return makeError(native.error()); }
    if (native->empty()) { return {}; }
    for (const auto& region : regions) {
        auto retained = retainBufferSlices(*this, {region.source, region.destination});
        if (!retained) { return retained; }
    }
    const VkDecompressMemoryInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DECOMPRESS_MEMORY_INFO_EXT,
        .decompressionMethod = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT,
        .regionCount = uint32_t(native->size()),
        .pRegions = native->data(),
    };
    impl_->device->functions.vkCmdDecompressMemoryEXT(impl_->commandBuffer, &info);
    return {};
}

Result<> CommandBuffer::validateDecompressionBuffers(std::span<const BufferDecompressionDesc> regions) const
{
    auto native = prepareDecompressionRegions(impl_.get(), recording_, regions);
    return native ? Result<>{} : makeError(native.error());
}

Result<> CommandBuffer::copyTexture(const TextureCopyDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_) ||
        desc.source == nullptr ||
        desc.source->impl_ == nullptr ||
        desc.destination == nullptr ||
        desc.destination->impl_ == nullptr ||
        desc.width == 0 ||
        desc.height == 0 ||
        desc.depth == 0 || desc.layerCount == 0 || desc.source->impl_->device != impl_->device ||
        desc.destination->impl_->device != impl_->device) {
        return makeError(Error::InvalidArgument);
    }

    const auto validRegion = [&](const TextureDesc& texture, uint32_t mip, uint32_t baseLayer) {
        return mip < texture.mipCount && mip < 32 && baseLayer < texture.layerCount &&
            desc.layerCount <= texture.layerCount - baseLayer &&
            desc.width <= std::max(1u, texture.width >> mip) &&
            desc.height <= std::max(1u, texture.height >> mip) &&
            desc.depth <= std::max(1u, texture.depth >> mip);
    };
    if (!validRegion(desc.source->desc(), desc.sourceMipLevel, desc.sourceBaseLayer) ||
        !validRegion(desc.destination->desc(), desc.destinationMipLevel, desc.destinationBaseLayer)) {
        return makeError(Error::InvalidArgument);
    }

    const VkImageAspectFlags sourceAspect = aspectForFormat(desc.source->impl_->desc.format);
    const VkImageAspectFlags destinationAspect = aspectForFormat(desc.destination->impl_->desc.format);
    if (sourceAspect != destinationAspect) {
        return makeError(Error::InvalidArgument);
    }

    VkImageCopy copyRegion{
        .srcSubresource = {
            .aspectMask = sourceAspect,
            .mipLevel = desc.sourceMipLevel,
            .baseArrayLayer = desc.sourceBaseLayer,
            .layerCount = desc.layerCount,
        },
        .srcOffset = {0, 0, 0},
        .dstSubresource = {
            .aspectMask = destinationAspect,
            .mipLevel = desc.destinationMipLevel,
            .baseArrayLayer = desc.destinationBaseLayer,
            .layerCount = desc.layerCount,
        },
        .dstOffset = {0, 0, 0},
        .extent = {desc.width, desc.height, desc.depth},
    };

    impl_->device->functions.vkCmdCopyImage(
        impl_->commandBuffer,
        desc.source->impl_->image,
        imageLayout(TextureLayout::TransferSource, impl_->device->vulkanCapabilities.unifiedImageLayouts),
        desc.destination->impl_->image,
        imageLayout(TextureLayout::TransferDestination, impl_->device->vulkanCapabilities.unifiedImageLayouts),
        1,
        &copyRegion);
    return {};
}

Result<> CommandBuffer::copyTextureToBuffer(const BufferTextureRegion& region)
{
    return copyBufferTexture(region, BufferTextureCopyDirection::ToBuffer);
}

Result<> CommandBuffer::copyBufferToTexture(const BufferTextureRegion& region)
{
    return copyBufferTexture(region, BufferTextureCopyDirection::ToTexture);
}

Result<> CommandBuffer::copyBufferTexture(const BufferTextureRegion& desc, BufferTextureCopyDirection direction)
{
    if (!validCommandRecording(impl_.get(), recording_) ||
        desc.texture == nullptr ||
        desc.texture->impl_ == nullptr ||
        desc.width == 0 ||
        desc.height == 0 ||
        desc.depth == 0 ||
        desc.layerCount == 0 || desc.texture->impl_->device != impl_->device ||
        !desc.buffer.validate(deviceIdentity(), direction == BufferTextureCopyDirection::ToTexture
            ? BufferUsageBits::TransferSource : BufferUsageBits::TransferDestination)) {
        return makeError(Error::InvalidArgument);
    }

    uint32_t bufferRowLength = 0;
    uint32_t bufferImageHeight = 0;
    if (!fillBufferImageLayout(desc, bufferRowLength, bufferImageHeight)) {
        return makeError(Error::InvalidArgument);
    }

    auto retained = retainResource(desc.buffer.retainAllocation());
    if (!retained) { return retained; }
    const VkDeviceMemoryImageCopyKHR copyRegion{
        .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
        .addressRange = {desc.buffer.deviceAddress(), desc.buffer.size()},
        .addressFlags = detail::BufferAddressCommandAccess::flags(desc.buffer),
        .addressRowLength = bufferRowLength,
        .addressImageHeight = bufferImageHeight,
        .imageSubresource = {
            .aspectMask = aspectForFormat(desc.texture->impl_->desc.format),
            .mipLevel = desc.mipLevel,
            .baseArrayLayer = desc.baseLayer,
            .layerCount = desc.layerCount,
        },
        .imageLayout = imageLayout(direction == BufferTextureCopyDirection::ToTexture
            ? TextureLayout::TransferDestination : TextureLayout::TransferSource,
            impl_->device->vulkanCapabilities.unifiedImageLayouts),
        .imageOffset = {desc.textureOffsetX, desc.textureOffsetY, desc.textureOffsetZ},
        .imageExtent = {desc.width, desc.height, desc.depth},
    };

    const VkCopyDeviceMemoryImageInfoKHR copyInfo{
        .sType = VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_IMAGE_INFO_KHR,
        .image = desc.texture->impl_->image,
        .regionCount = 1,
        .pRegions = &copyRegion,
    };
    if (direction == BufferTextureCopyDirection::ToTexture) {
        impl_->device->functions.vkCmdCopyMemoryToImageKHR(impl_->commandBuffer, &copyInfo);
    } else {
        impl_->device->functions.vkCmdCopyImageToMemoryKHR(impl_->commandBuffer, &copyInfo);
    }
    return {};
}

void CommandBuffer::hostWriteBarrier()
{
    if (!validCommandRecording(impl_.get(), recording_)) {
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
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, dependency);
}

Result<> CommandBuffer::clearColorTexture(Texture& texture, TextureLayout layout, const ColorValue& color)
{
    if (!validCommandRecording(impl_.get(), recording_) || texture.impl_ == nullptr || texture.impl_->image == VK_NULL_HANDLE ||
        texture.impl_->device != impl_->device ||
        (layout != TextureLayout::TransferDestination && layout != TextureLayout::General)) {
        return makeError(Error::InvalidArgument);
    }
    const TextureDesc& desc = texture.impl_->desc;
    if (aspectForFormat(desc.format) != VK_IMAGE_ASPECT_COLOR_BIT) {
        return makeError(Error::InvalidArgument);
    }

    const VkClearColorValue clearValue{{color.r, color.g, color.b, color.a}};
    const VkImageSubresourceRange range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = desc.mipCount,
        .baseArrayLayer = 0,
        .layerCount = desc.layerCount,
    };
    impl_->device->functions.vkCmdClearColorImage(
        impl_->commandBuffer,
        texture.impl_->image,
        imageLayout(layout, impl_->device->vulkanCapabilities.unifiedImageLayouts),
        &clearValue,
        1,
        &range);
    return {};
}

Result<> CommandBuffer::useNativeTextureView(TextureView& view)
{
    if (!validCommandRecording(impl_.get(), recording_) || view.deviceIdentity() != deviceIdentity()) { return makeError(Error::InvalidArgument); }
    auto result = view.impl_->materialize();
    return result ? retainResource(view.impl_) : result;
}

Result<> CommandBuffer::beginRendering(const RenderingDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT)) {
        return makeError(Error::InvalidArgument);
    }
    if ((desc.colorAttachments.size() > UINT32_MAX) ||
        (desc.depthStencilAttachment && !desc.depthStencilAttachment->view)) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<VkRenderingAttachmentInfo> colorAttachments;
    colorAttachments.reserve(desc.colorAttachments.size());
    VkRenderingAttachmentInfo depthAttachment{};
    const VkRenderingAttachmentInfo* depthAttachmentPtr = nullptr;

    for (uint32_t index = 0; index < desc.colorAttachments.size(); ++index) {
        const RenderingAttachmentDesc& attachment = desc.colorAttachments[index];
        if (attachment.view == nullptr || attachment.view->impl_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        auto result = useNativeTextureView(*attachment.view);
        if (!result) { return result; }
        const ColorValue& clear = attachment.clearColor;
        colorAttachments.push_back({
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = attachment.view->impl_->view,
            .imageLayout = imageLayout(attachment.layout, impl_->device->vulkanCapabilities.unifiedImageLayouts),
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
            auto result = useNativeTextureView(*attachment.view);
            if (!result) { return result; }
            depthAttachment = {
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = attachment.view->impl_->view,
                .imageLayout = imageLayout(attachment.layout, impl_->device->vulkanCapabilities.unifiedImageLayouts),
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
    impl_->device->functions.vkCmdBeginRendering(impl_->commandBuffer, &renderingInfo);
    observeRHICommand(RHICommandEvent::BeginRendering, this);
    return {};
}

void CommandBuffer::clearColorAttachment(uint32_t attachmentIndex, const ColorValue& color, const Rect& rect)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT)) {
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

    impl_->device->functions.vkCmdClearAttachments(impl_->commandBuffer, 1, &attachment, 1, &clearRect);
}

void CommandBuffer::endRendering()
{
    if (validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT)) {
        impl_->device->functions.vkCmdEndRendering(impl_->commandBuffer);
        observeRHICommand(RHICommandEvent::EndRendering, this);
    }
}

Result<> CommandBuffer::setViewport(const Viewport& viewport)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT)) {
        return makeError(Error::InvalidArgument);
    }

    if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) ||
        !std::isfinite(viewport.width) || !std::isfinite(viewport.height) ||
        !std::isfinite(viewport.minDepth) || !std::isfinite(viewport.maxDepth) ||
        viewport.width <= 0 || viewport.height == 0 || viewport.minDepth < 0 || viewport.minDepth > 1 ||
        viewport.maxDepth < 0 || viewport.maxDepth > 1) {
        return makeError(Error::InvalidArgument);
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
    impl_->device->functions.vkCmdSetViewport(impl_->commandBuffer, 0, 1, &vkViewport);
    if (impl_->currentGraphicsShaderObjectBound && impl_->device->functions.vkCmdSetViewportWithCountEXT != nullptr) {
        impl_->device->functions.vkCmdSetViewportWithCountEXT(impl_->commandBuffer, 1, &vkViewport);
    }
    return {};
}

void CommandBuffer::setScissor(const Rect& scissor)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT)) {
        return;
    }

    VkRect2D vkScissor{
        .offset = {scissor.x, scissor.y},
        .extent = {scissor.width, scissor.height},
    };
    impl_->currentScissor = scissor;
    impl_->hasCurrentScissor = true;
    impl_->device->functions.vkCmdSetScissor(impl_->commandBuffer, 0, 1, &vkScissor);
    if (impl_->currentGraphicsShaderObjectBound && impl_->device->functions.vkCmdSetScissorWithCountEXT != nullptr) {
        impl_->device->functions.vkCmdSetScissorWithCountEXT(impl_->commandBuffer, 1, &vkScissor);
    }
}

void CommandBuffer::setDepthStencilState(const DepthStencilState& state)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT)) {
        return;
    }

    if (impl_->device->functions.vkCmdSetDepthTestEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthTestEnableEXT(
            impl_->commandBuffer,
            state.depthTestEnable ? VK_TRUE : VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetDepthWriteEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthWriteEnableEXT(
            impl_->commandBuffer,
            state.depthWriteEnable ? VK_TRUE : VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetDepthCompareOpEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthCompareOpEXT(
            impl_->commandBuffer,
            toVkCompareOp(state.depthCompareOp));
    }
}

namespace {

struct GraphicsShaderObjectStages {
    std::array<VkShaderStageFlagBits, 7> stages{VK_SHADER_STAGE_VERTEX_BIT,
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    uint32_t count = 5;
};

GraphicsShaderObjectStages graphicsShaderObjectStages(const DeviceCapabilities& capabilities)
{
    GraphicsShaderObjectStages result;
    // Enabled task/mesh stages must have an explicit binding, including null
    // bindings when executing a conventional vertex/fragment program.
    if (capabilities.taskShader) { result.stages[result.count++] = VK_SHADER_STAGE_TASK_BIT_EXT; }
    if (capabilities.meshShader) { result.stages[result.count++] = VK_SHADER_STAGE_MESH_BIT_EXT; }
    return result;
}

void clearGraphicsShaderObjects(detail::CommandBufferImpl& commandBuffer)
{
    if (!commandBuffer.currentGraphicsShaderObjectBound ||
        commandBuffer.device == nullptr ||
        !commandBuffer.device->capabilities.shaderObject ||
        commandBuffer.device->functions.vkCmdBindShadersEXT == nullptr) {
        commandBuffer.currentGraphicsShaderObjectBound = false;
        commandBuffer.currentGraphicsShaderObjectUsesBindlessHeap = false;
        return;
    }

    const auto stages = graphicsShaderObjectStages(commandBuffer.device->capabilities);
    commandBuffer.device->functions.vkCmdBindShadersEXT(
        commandBuffer.commandBuffer,
        stages.count,
        stages.stages.data(),
        nullptr);
    commandBuffer.currentGraphicsShaderObjectBound = false;
    commandBuffer.currentGraphicsShaderObjectUsesBindlessHeap = false;
}

// Descriptor indices in the payload are already relative to the bound heap.
// Push the exact caller ABI from byte zero; no backend header is prepended.
bool validPushData(const detail::CommandBufferImpl& commandBuffer, const void* data, uint32_t byteSize)
{
    return (!byteSize || data) && !(byteSize & 3u) &&
        byteSize <= commandBuffer.device->descriptorHeapWriter.maxPushDataSize();
}

void pushCurrentBindlessData(detail::CommandBufferImpl& commandBuffer)
{
    const auto& payload = commandBuffer.currentBindlessUserData;
    const bool needsDescriptorHeapPush =
        commandBuffer.currentGraphicsPipelineUsesBindlessHeap ||
        commandBuffer.currentComputePipelineUsesBindlessHeap ||
        commandBuffer.currentGraphicsShaderObjectUsesBindlessHeap;
    if (needsDescriptorHeapPush &&
        !payload.empty() &&
        commandBuffer.device != nullptr &&
        commandBuffer.device->bindlessDescriptorHeapEnabled &&
        commandBuffer.device->descriptorHeapWriter.maxPushDataSize() >= payload.size() &&
        commandBuffer.device->functions.vkCmdPushDataEXT != nullptr) {
        const VkPushDataInfoEXT pushInfo{
            .sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
            .offset = 0,
            .data = {
                .address = payload.data(),
                .size = payload.size(),
            },
        };
        commandBuffer.device->functions.vkCmdPushDataEXT(commandBuffer.commandBuffer, &pushInfo);
    }
}

} // namespace

Result<> CommandBuffer::bindExecution(const PreparedExecution& execution)
{
    return bindExecutionImpl(execution, nullptr, 0, false);
}

Result<> CommandBuffer::bindExecutionImpl(
    const PreparedExecution& execution,
    const void* data,
    uint32_t byteSize,
    bool replaceData)
{
    if (!validCommandRecording(impl_.get(), recording_, execution.kind() == ExecutionKind::Compute
            ? VK_QUEUE_COMPUTE_BIT : VK_QUEUE_GRAPHICS_BIT) ||
        !execution.valid() || execution.deviceIdentity() != deviceIdentity() ||
        (execution.shaders_ && execution.raster_.colorAttachmentCount > 8) ||
        (replaceData && !validPushData(*impl_, data, byteSize))) { return makeError(Error::InvalidArgument); }
    if (execution.compute_) {
        auto result = retainResource(execution.compute_);
        if (!result) { return result; }
        const auto& pipeline = *execution.compute_;
        impl_->device->functions.vkCmdBindPipeline(impl_->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        impl_->currentComputePipeline = pipeline.pipeline;
        impl_->currentComputePipelineLayout = pipeline.layout;
        impl_->currentComputePipelineUsesBindlessHeap = pipeline.usesBindlessHeap;
    } else if (execution.graphics_) {
        auto result = retainResource(execution.graphics_);
        if (!result) { return result; }
        const auto& pipeline = *execution.graphics_;
        clearGraphicsShaderObjects(*impl_);
        impl_->device->functions.vkCmdBindPipeline(impl_->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
        impl_->currentGraphicsPipelineLayout = pipeline.layout;
        impl_->currentGraphicsPipelineUsesBindlessHeap = pipeline.usesBindlessHeap;
    } else {
        if (!impl_->device->capabilities.shaderObject || !impl_->device->functions.vkCmdBindShadersEXT) { return makeError(Error::Unsupported); }
        auto result = retainResource(execution.shaders_);
        if (!result) { return result; }
        const auto& program = *execution.shaders_;
        const auto stages = graphicsShaderObjectStages(impl_->device->capabilities);
        const std::array<VkShaderEXT, 7> shaders{program.vertexShader, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, program.fragmentShader};
        impl_->device->functions.vkCmdBindShadersEXT(impl_->commandBuffer, stages.count, stages.stages.data(), shaders.data());
        impl_->currentGraphicsPipelineLayout = VK_NULL_HANDLE;
        impl_->currentGraphicsPipelineUsesBindlessHeap = false;
        impl_->currentGraphicsShaderObjectBound = true;
        impl_->currentGraphicsShaderObjectUsesBindlessHeap = program.usesBindlessHeap;
        setGraphicsShaderObjectState();
        setDepthStencilState(execution.raster_.depthStencil);
        impl_->device->functions.vkCmdSetCullModeEXT(impl_->commandBuffer, toVkCullMode(execution.raster_.rasterization.cullMode));
        impl_->device->functions.vkCmdSetFrontFaceEXT(impl_->commandBuffer, toVkFrontFace(execution.raster_.rasterization.frontFace));
        for (uint32_t i = 0; i < execution.raster_.colorAttachmentCount; ++i) {
            const VkBool32 blend = VK_FALSE;
            const VkColorBlendEquationEXT equation{VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
                VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};
            const VkColorComponentFlags mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            impl_->device->functions.vkCmdSetColorBlendEnableEXT(impl_->commandBuffer, i, 1, &blend);
            impl_->device->functions.vkCmdSetColorBlendEquationEXT(impl_->commandBuffer, i, 1, &equation);
            impl_->device->functions.vkCmdSetColorWriteMaskEXT(impl_->commandBuffer, i, 1, &mask);
        }

    }
    if (replaceData) {
        impl_->currentBindlessUserData.resize(byteSize);
        if (byteSize) { std::memcpy(impl_->currentBindlessUserData.data(), data, byteSize); }
    }
    if (impl_->currentBindlessHeap) { pushCurrentBindlessData(*impl_); }
    return {};
}

Result<> CommandBuffer::bindExecution(const PreparedExecution& execution, const void* data, uint32_t byteSize)
{
    return bindExecutionImpl(execution, data, byteSize, true);
}

void CommandBuffer::setGraphicsShaderObjectState()
{
    if (impl_ == nullptr ||
        impl_->device == nullptr ||
        !impl_->device->capabilities.shaderObject) {
        return;
    }

    if (impl_->device->functions.vkCmdSetVertexInputEXT != nullptr) {
        impl_->device->functions.vkCmdSetVertexInputEXT(impl_->commandBuffer, 0, nullptr, 0, nullptr);
    }
    if (impl_->device->functions.vkCmdSetPrimitiveTopologyEXT != nullptr) {
        impl_->device->functions.vkCmdSetPrimitiveTopologyEXT(impl_->commandBuffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    }
    if (impl_->device->functions.vkCmdSetPrimitiveRestartEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetPrimitiveRestartEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetRasterizerDiscardEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetRasterizerDiscardEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetPolygonModeEXT != nullptr) {
        impl_->device->functions.vkCmdSetPolygonModeEXT(impl_->commandBuffer, VK_POLYGON_MODE_FILL);
    }
    if (impl_->device->functions.vkCmdSetCullModeEXT != nullptr) {
        impl_->device->functions.vkCmdSetCullModeEXT(impl_->commandBuffer, VK_CULL_MODE_NONE);
    }
    if (impl_->device->functions.vkCmdSetFrontFaceEXT != nullptr) {
        impl_->device->functions.vkCmdSetFrontFaceEXT(impl_->commandBuffer, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    }
    if (impl_->device->functions.vkCmdSetDepthClampEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthClampEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetDepthBiasEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthBiasEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    impl_->device->functions.vkCmdSetLineWidth(impl_->commandBuffer, 1.0f);
    if (impl_->device->functions.vkCmdSetRasterizationSamplesEXT != nullptr) {
        impl_->device->functions.vkCmdSetRasterizationSamplesEXT(impl_->commandBuffer, VK_SAMPLE_COUNT_1_BIT);
    }
    if (impl_->device->functions.vkCmdSetSampleMaskEXT != nullptr) {
        const VkSampleMask sampleMask = 0xffffffffu;
        impl_->device->functions.vkCmdSetSampleMaskEXT(impl_->commandBuffer, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
    }
    if (impl_->device->functions.vkCmdSetAlphaToCoverageEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetAlphaToCoverageEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetAlphaToOneEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetAlphaToOneEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetDepthTestEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthTestEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetDepthWriteEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthWriteEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetDepthCompareOpEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthCompareOpEXT(impl_->commandBuffer, VK_COMPARE_OP_ALWAYS);
    }
    if (impl_->device->functions.vkCmdSetDepthBoundsTestEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetDepthBoundsTestEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetStencilTestEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetStencilTestEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetColorBlendEnableEXT != nullptr) {
        const VkBool32 blendEnable = VK_FALSE;
        impl_->device->functions.vkCmdSetColorBlendEnableEXT(impl_->commandBuffer, 0, 1, &blendEnable);
    }
    if (impl_->device->functions.vkCmdSetColorBlendEquationEXT != nullptr) {
        const VkColorBlendEquationEXT blendEquation{
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
        };
        impl_->device->functions.vkCmdSetColorBlendEquationEXT(impl_->commandBuffer, 0, 1, &blendEquation);
    }
    if (impl_->device->functions.vkCmdSetColorWriteMaskEXT != nullptr) {
        const VkColorComponentFlags colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        impl_->device->functions.vkCmdSetColorWriteMaskEXT(impl_->commandBuffer, 0, 1, &colorWriteMask);
    }
    if (impl_->device->functions.vkCmdSetLogicOpEnableEXT != nullptr) {
        impl_->device->functions.vkCmdSetLogicOpEnableEXT(impl_->commandBuffer, VK_FALSE);
    }
    if (impl_->device->functions.vkCmdSetLogicOpEXT != nullptr) {
        impl_->device->functions.vkCmdSetLogicOpEXT(impl_->commandBuffer, VK_LOGIC_OP_COPY);
    }

    if (impl_->hasCurrentViewport && impl_->device->functions.vkCmdSetViewportWithCountEXT != nullptr) {
        const VkViewport viewport{
            .x = impl_->currentViewport.x,
            .y = impl_->currentViewport.y,
            .width = impl_->currentViewport.width,
            .height = impl_->currentViewport.height,
            .minDepth = impl_->currentViewport.minDepth,
            .maxDepth = impl_->currentViewport.maxDepth,
        };
        impl_->device->functions.vkCmdSetViewportWithCountEXT(impl_->commandBuffer, 1, &viewport);
    }
    if (impl_->hasCurrentScissor && impl_->device->functions.vkCmdSetScissorWithCountEXT != nullptr) {
        const VkRect2D scissor{
            .offset = {impl_->currentScissor.x, impl_->currentScissor.y},
            .extent = {impl_->currentScissor.width, impl_->currentScissor.height},
        };
        impl_->device->functions.vkCmdSetScissorWithCountEXT(impl_->commandBuffer, 1, &scissor);
    }
}

Result<> CommandBuffer::bindBindlessHeap(BindlessHeap& heap)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) ||
        !heap.impl_ || heap.impl_->device != impl_->device) { return makeError(Error::InvalidArgument); }
    if (impl_->currentBindlessHeap == heap.impl_.get()) { return {}; }
    impl_->currentBindlessHeap = heap.impl_.get();
    heap.impl_->heap.bind(impl_->commandBuffer, heap.impl_->samplerHeap.address, heap.impl_->resourceHeap.address);
    pushCurrentBindlessData(*impl_);
    return {};
}

Result<> CommandBuffer::pushBindlessData(const void* data, uint32_t byteSize)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) ||
        !validPushData(*impl_, data, byteSize)) { return makeError(Error::InvalidArgument); }
    impl_->currentBindlessUserData.resize(byteSize);
    if (byteSize) { std::memcpy(impl_->currentBindlessUserData.data(), data, byteSize); }
    if (impl_->currentBindlessHeap) { pushCurrentBindlessData(*impl_); }
    return {};
}

Result<> CommandBuffer::recordIsolatedCompute(const std::function<Result<>()>& record)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT) || !record) {
        return makeError(Error::InvalidArgument);
    }
    const auto pipeline = impl_->currentComputePipeline;
    const auto layout = impl_->currentComputePipelineLayout;
    const auto usesHeap = impl_->currentComputePipelineUsesBindlessHeap;
    auto* heap = impl_->currentBindlessHeap;
    auto data = impl_->currentBindlessUserData;
    const auto restore = [&] {
        impl_->currentComputePipeline = pipeline;
        impl_->currentComputePipelineLayout = layout;
        impl_->currentComputePipelineUsesBindlessHeap = usesHeap;
        impl_->currentBindlessHeap = heap;
        impl_->currentBindlessUserData = std::move(data);
        if (pipeline) { impl_->device->functions.vkCmdBindPipeline(impl_->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline); }
        if (heap) {
            heap->heap.bind(impl_->commandBuffer, heap->samplerHeap.address, heap->resourceHeap.address);
            pushCurrentBindlessData(*impl_);
        }
    };
    try {
        const auto result = record();
        restore();
        return result;
    } catch (...) { restore(); throw; }
}

Result<> CommandBuffer::draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT)) {
        return makeError(Error::InvalidArgument);
    }
    if (!vertexCount || !instanceCount) { return {}; }
    impl_->device->functions.vkCmdDraw(impl_->commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
    observeRHICommand(RHICommandEvent::Draw, this);
    return {};
}

Result<> CommandBuffer::drawMeshTasks(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT)) {
        return makeError(Error::InvalidArgument);
    }
    if (!groupCountX || !groupCountY || !groupCountZ) { return {}; }
#ifdef VK_EXT_mesh_shader
    if (!impl_->device->capabilities.meshShader || impl_->device->functions.vkCmdDrawMeshTasksEXT == nullptr) {
        return makeError(Error::Unsupported);
    }
    impl_->device->functions.vkCmdDrawMeshTasksEXT(impl_->commandBuffer, groupCountX, groupCountY, groupCountZ);
    observeRHICommand(RHICommandEvent::Draw, this);
#else
    return makeError(Error::Unsupported);
#endif
    return {};
}

Result<> CommandBuffer::drawMeshTasksIndirect(const BufferSlice& arguments)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_GRAPHICS_BIT) ||
        !arguments.validate(deviceIdentity(), BufferUsageBits::Indirect, 4, sizeof(VkDrawMeshTasksIndirectCommandEXT))) {
        return makeError(Error::InvalidArgument);
    }
#ifdef VK_EXT_mesh_shader
    if (!impl_->device->capabilities.meshShader || impl_->device->functions.vkCmdDrawMeshTasksIndirect2EXT == nullptr) {
        return makeError(Error::Unsupported);
    }
    auto retained = retainResource(arguments.retainAllocation());
    if (!retained) { return retained; }
    const VkDrawIndirect2InfoKHR info{
        .sType = VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR,
        .addressRange = {arguments.deviceAddress(), sizeof(VkDrawMeshTasksIndirectCommandEXT), 0},
        .addressFlags = detail::BufferAddressCommandAccess::flags(arguments),
        .drawCount = 1,
    };
    impl_->device->functions.vkCmdDrawMeshTasksIndirect2EXT(impl_->commandBuffer, &info);
    observeRHICommand(RHICommandEvent::Draw, this);
#else
    return makeError(Error::Unsupported);
#endif
    return {};
}

Result<> CommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT)) { return makeError(Error::InvalidArgument); }
    if (!groupCountX || !groupCountY || !groupCountZ) { return {}; }
    const auto& limits = impl_->device->physicalProperties.core.limits.maxComputeWorkGroupCount;
    if (groupCountX > limits[0] || groupCountY > limits[1] || groupCountZ > limits[2]) { return makeError(Error::InvalidArgument); }
    impl_->device->functions.vkCmdDispatch(impl_->commandBuffer, groupCountX, groupCountY, groupCountZ);
    observeRHICommand(RHICommandEvent::Dispatch, this);
    return {};
}

Result<> CommandBuffer::dispatchIndirect(const BufferSlice& arguments)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT) ||
        !arguments.validate(deviceIdentity(), BufferUsageBits::Indirect, 4, sizeof(VkDispatchIndirectCommand))) {
        return makeError(Error::InvalidArgument);
    }
    auto result = retainResource(arguments.retainAllocation());
    if (!result) { return result; }
    const VkDispatchIndirect2InfoKHR info{
        .sType = VK_STRUCTURE_TYPE_DISPATCH_INDIRECT_2_INFO_KHR,
        .addressRange = {arguments.deviceAddress(), sizeof(VkDispatchIndirectCommand)},
        .addressFlags = detail::BufferAddressCommandAccess::flags(arguments),
    };
    impl_->device->functions.vkCmdDispatchIndirect2KHR(impl_->commandBuffer, &info);
    observeRHICommand(RHICommandEvent::Dispatch, this);
    return {};
}

Result<> CommandBuffer::buildRayTracingAccelerationStructure(
    const RayTracingAccelerationStructureBuildDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT) ||
        desc.destination == nullptr ||
        desc.destination->impl_ == nullptr || !desc.destination->valid() ||
        desc.destination->impl_->device != impl_->device ||
        desc.destination->desc().topLevelBackend != RayTracingTopLevelBackend::Standard ||
        !validCommandSlice(*impl_, desc.scratchBuffer, BufferUsageBits::Storage)) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->device->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
    const auto queueCanAccess = [&](const Buffer& buffer) {
        const auto families = detail::queueFamiliesForAccess(*impl_->device, buffer.desc().queueAccess);
        return std::find(families.begin(), families.end(), impl_->queueFamilyIndex) != families.end();
    };
    if (!queueCanAccess(*desc.destination->impl_->storage)) {
        return makeError(Error::InvalidArgument);
    }

    const RayTracingAccelerationStructureDesc& destinationDesc =
        desc.destination->impl_->desc;
    if (desc.mode == RayTracingAccelerationStructureBuildMode::Update) {
        if (!hasFlag(
                destinationDesc.buildFlags,
                RayTracingAccelerationStructureBuildFlags::AllowUpdate) ||
            desc.source == nullptr || desc.source->impl_ == nullptr ||
            !desc.source->valid() || desc.source->impl_->device != impl_->device ||
            desc.source->desc().topLevelBackend != RayTracingTopLevelBackend::Standard ||
            desc.source->impl_->desc.type != destinationDesc.type ||
            !queueCanAccess(*desc.source->impl_->storage)) {
            return makeError(Error::InvalidArgument);
        }
    } else if (desc.source != nullptr) {
        return makeError(Error::InvalidArgument);
    }

    auto* plan = desc.plan ? desc.plan->impl_.get() : nullptr;
    if (desc.plan && (!desc.plan->valid() || plan->device != impl_->device ||
            desc.mode != RayTracingAccelerationStructureBuildMode::Build ||
            destinationDesc.type != RayTracingAccelerationStructureType::BottomLevel ||
            plan->flags != destinationDesc.buildFlags ||
            desc.destination->impl_->coverageBuildIdentity != plan->identity ||
            plan->recorded.load(std::memory_order_acquire) || !desc.geometries.empty())) {
        return makeError(Error::InvalidArgument);
    }
    if (!plan && (!desc.destination->impl_->coverageDependencies.empty() ||
            (desc.source && !desc.source->impl_->coverageDependencies.empty()))) {
        return makeError(Error::Unsupported);
    }
    const std::span<const RayTracingTriangleGeometryDesc> sourceGeometries =
        plan ? std::span<const RayTracingTriangleGeometryDesc>(plan->geometries) : desc.geometries;
    if (sourceGeometries.size() > UINT32_MAX || std::any_of(sourceGeometries.begin(), sourceGeometries.end(),
            [](const auto& geometry) { return geometry.coverage != nullptr; })) { return makeError(Error::InvalidArgument); }
    std::vector<VkAccelerationStructureTrianglesOpacityMicromapKHR> attachments(sourceGeometries.size());
    std::vector<VkAccelerationStructureTrianglesOpacityMicromapEXT> extAttachments(sourceGeometries.size());
    std::vector<std::vector<VkMicromapUsageEXT>> extAttachmentUsages(sourceGeometries.size());
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    if (destinationDesc.type == RayTracingAccelerationStructureType::BottomLevel) {
        if (sourceGeometries.empty() || sourceGeometries.size() > UINT32_MAX ||
            desc.instanceBuffer.valid() || desc.instanceCount != 0) {
            return makeError(Error::InvalidArgument);
        }
        geometries.reserve(sourceGeometries.size());
        ranges.reserve(sourceGeometries.size());
        for (uint32_t index = 0; index < sourceGeometries.size(); ++index) {
            const RayTracingTriangleGeometryDesc& source = sourceGeometries[index];
            if (!validTriangleGeometry(impl_->device, source) ||
                !queueCanAccessBuffer(*impl_, source.vertexBuffer.allocationDesc()) ||
                (source.indexType != RayTracingIndexType::None && !queueCanAccessBuffer(*impl_, source.indexBuffer.allocationDesc()))) {
                return makeError(Error::InvalidArgument);
            }
            const VkDeviceAddress vertexAddress = source.vertexBuffer.deviceAddress();
            const VkFormat vertexFormat = toVkFormat(source.vertexFormat);
            const VkDeviceAddress indexAddress = source.indexType == RayTracingIndexType::None ? 0 : source.indexBuffer.deviceAddress();

            const void* opacityAttachment = nullptr;
            if (plan && plan->coverage[index].resource) {
                const auto& coverage = plan->coverage[index];
                if (coverage.resource->micromap != VK_NULL_HANDLE) {
                    VkDeviceAddress identityIndexAddress = 0;
                    const auto indices = detail::ensureMicromapIdentityIndices(*coverage.resource,
                        source.primitiveCount, identityIndexAddress);
                    if (!indices) { return indices; }
                    const auto attached = makeExtMicromapAttachment(source.primitiveCount,
                        coverage.baked.usages, coverage.resource->micromap,
                        extAttachmentUsages[index], extAttachments[index], identityIndexAddress);
                    if (!attached) { return attached; }
                    opacityAttachment = &extAttachments[index];
                } else {
                    attachments[index] = {
                        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_KHR,
                        .indexType = VK_INDEX_TYPE_NONE_KHR,
                        .micromap = coverage.resource->accelerationStructure,
                    };
                    opacityAttachment = &attachments[index];
                }
            }
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .pNext = opacityAttachment,
                .vertexFormat = vertexFormat,
                .vertexData = {.deviceAddress = vertexAddress},
                .vertexStride = source.vertexStride,
                .maxVertex = source.vertexCount - 1,
                .indexType = toVkRayTracingIndexType(source.indexType),
                .indexData = {.deviceAddress = indexAddress},
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
        if (destinationDesc.type != RayTracingAccelerationStructureType::TopLevel || !sourceGeometries.empty() ||
            !desc.instanceCount || !validCommandSlice(*impl_, desc.instanceBuffer,
                BufferUsageBits::AccelerationStructureBuildInput, uint64_t(desc.instanceCount) * sizeof(VkAccelerationStructureInstanceKHR), 16)) {
            return makeError(Error::InvalidArgument);
        }
        const VkDeviceAddress instanceAddress = desc.instanceBuffer.deviceAddress();
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

    const uint64_t scratchAlignment = rayTracingScratchAlignment(*impl_->device);
    auto scratch = alignedBuildScratch(*impl_, desc.scratchBuffer, scratchAlignment, 1);
    if (!scratch) { return makeError(scratch.error()); }
    const VkDeviceAddress scratchAddress = scratch->deviceAddress();

    const auto retainBuildResources = [&]() -> Result<> {
        auto result = retainResource(desc.destination->retainAllocation());
        if (result) { result = retainResource(desc.scratchBuffer.retainAllocation()); }
        if (result && desc.source) { result = retainResource(desc.source->retainAllocation()); }
        if (result && desc.instanceBuffer.valid()) { result = retainResource(desc.instanceBuffer.retainAllocation()); }
        for (const auto& geometry : sourceGeometries) {
            if (result) { result = retainResource(geometry.vertexBuffer.retainAllocation()); }
            if (result && geometry.indexType != RayTracingIndexType::None) {
                result = retainResource(geometry.indexBuffer.retainAllocation());
            }
        }
        return result;
    };

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
    impl_->device->functions.vkGetAccelerationStructureBuildSizesKHR(
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
        requiredScratchSize > scratch->size() ||
        (plan && plan->sizes.buildScratchSize > scratch->size()) ||
        sizes.accelerationStructureSize > destinationDesc.size) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers;
    rangePointers.reserve(ranges.size());
    for (const VkAccelerationStructureBuildRangeInfoKHR& range : ranges) {
        rangePointers.push_back(&range);
    }
    const auto retained = retainBuildResources();
    if (!retained) { return retained; }
    std::vector<std::shared_ptr<detail::MicromapBufferAllocation>> uploads;
    std::vector<NativeMicromapBuild> nativeMicromapBuilds;
    if (plan) {
        uploads.resize(plan->coverage.size());
        nativeMicromapBuilds.resize(plan->coverage.size());
        for (size_t index = 0; index < plan->coverage.size(); ++index) {
            const auto& coverage = plan->coverage[index];
            if (!coverage.resource) { continue; }
            auto upload = createMicromapBuildUpload(*impl_->device, coverage.baked);
            if (!upload) { return makeError(upload.error()); }
            uploads[index] = std::move(*upload);
            prepareNativeMicromapBuild(nativeMicromapBuilds[index], impl_->device->opacityMicromapExt,
                coverage, *uploads[index], scratchAddress);
        }
        for (const auto& upload : uploads) {
            if (!upload) { continue; }
            const auto retainedUpload = retainResource(upload);
            if (!retainedUpload) { return retainedUpload; }
        }
        bool expected = false;
        if (!plan->recorded.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return makeError(Error::InvalidArgument);
        }
        // All validation, allocation and retention precede the first native command.
        // The plan is consumed even if its containing recording is later abandoned.
        for (size_t index = 0; index < plan->coverage.size(); ++index) {
            if (uploads[index]) { recordPreparedMicromapBuild(*impl_, nativeMicromapBuilds[index]); }
        }
    }
    impl_->device->functions.vkCmdBuildAccelerationStructuresKHR(
        impl_->commandBuffer,
        1,
        &buildInfo,
        rangePointers.data());

    if (desc.graphManagedSynchronization) { return {}; }

    const VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : 0),
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_ACCESS_2_SHADER_READ_BIT : 0),
    };
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, dependency);
    return {};
}

Result<> CommandBuffer::compactRayTracingAccelerationStructure(
    RayTracingAccelerationStructure& source,
    RayTracingAccelerationStructure& destination)
{
    // Coverage dependencies are published during recording. A fresh raw
    // destination prevents replacement of owners needed by prior GPU work.
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT) ||
        source.impl_ == nullptr || destination.impl_ == nullptr ||
        source.impl_.get() == destination.impl_.get() ||
        source.impl_->device != impl_->device || destination.impl_->device != impl_->device ||
        !source.valid() || !destination.valid() ||
        source.impl_->accelerationStructure == VK_NULL_HANDLE ||
        destination.impl_->accelerationStructure == VK_NULL_HANDLE ||
        destination.impl_->coverageBuildIdentity ||
        !destination.impl_->coverageDependencies.empty() ||
        source.impl_->desc.type != destination.impl_->desc.type ||
        source.impl_->desc.buildFlags != destination.impl_->desc.buildFlags ||
        !hasFlag(
            source.impl_->desc.buildFlags,
            RayTracingAccelerationStructureBuildFlags::AllowCompaction)) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->device->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    auto retained = retainResource(source.retainAllocation());
    if (retained) { retained = retainResource(destination.retainAllocation()); }
    if (!retained) { return retained; }
    destination.impl_->coverageDependencies = source.impl_->coverageDependencies;
    destination.impl_->coverageBuildIdentity = source.impl_->coverageBuildIdentity;
    destination.impl_->coverageStats = source.impl_->coverageStats;

    const VkMemoryBarrier2 beforeCopyBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    const VkDependencyInfo beforeCopyDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &beforeCopyBarrier,
    };
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, beforeCopyDependency);

    const VkCopyAccelerationStructureInfoKHR copyInfo{
        .sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
        .src = source.impl_->accelerationStructure,
        .dst = destination.impl_->accelerationStructure,
        .mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR,
    };
    impl_->device->functions.vkCmdCopyAccelerationStructureKHR(impl_->commandBuffer, &copyInfo);

    const VkMemoryBarrier2 afterCopyBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : 0),
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_ACCESS_2_SHADER_READ_BIT : 0),
    };
    const VkDependencyInfo afterCopyDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &afterCopyBarrier,
    };
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, afterCopyDependency);
    return {};
}

Result<> CommandBuffer::buildClusterAccelerationStructureTriangles(
    const ClusterAccelerationStructureTriangleBuildDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT)) {
        return makeError(Error::InvalidArgument);
    }
#ifndef VK_NV_cluster_acceleration_structure
    (void)desc;
    return makeError(Error::Unsupported);
#else
    if (!impl_->device->capabilities.clusterAccelerationStructure ||
        impl_->device->functions.vkCmdBuildClusterAccelerationStructureIndirectNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    if (desc.clusters.empty() || desc.clusters.size() > UINT32_MAX ||
        !desc.maxClusterTriangleCount || !desc.maxClusterVertexCount ||
        !desc.maxClusterUniqueGeometryCount || desc.vertexFormat == Format::Unknown) {
        return makeError(Error::InvalidArgument);
    }
    const uint64_t buildInfoBytes = uint64_t(desc.clusters.size()) *
        sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV);
    const uint64_t destinationAddressBytes = uint64_t(desc.clusters.size()) * sizeof(uint64_t);
    if (!validCommandSlice(*impl_, desc.buildInfoBuffer, BufferUsageBits::AccelerationStructureBuildInput, buildInfoBytes, 8) ||
        !validCommandSlice(*impl_, desc.destinationAddressBuffer, BufferUsageBits::AccelerationStructureStorage, destinationAddressBytes, 8) ||
        (desc.destinationSizeBuffer.valid() && !validCommandSlice(*impl_, desc.destinationSizeBuffer,
            BufferUsageBits::AccelerationStructureStorage, uint64_t(desc.clusters.size()) * sizeof(uint32_t), 4))) {
        return makeError(Error::InvalidArgument);
    }
    const auto& limits = impl_->device->physicalProperties.cluster;
    const auto vertexInfo = formatInfo(desc.vertexFormat);
    if (!vertexInfo.bytesPerBlock || vertexInfo.blockExtent != 1) { return makeError(Error::InvalidArgument); }

    std::vector<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> buildInfos(
        desc.clusters.size());
    std::vector<uint64_t> destinationAddresses(desc.clusters.size());
    uint64_t totalTriangleCount = 0;
    uint64_t totalVertexCount = 0;
    for (uint32_t index = 0; index < desc.clusters.size(); ++index) {
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
            source.vertexBufferStride < vertexInfo.bytesPerBlock) {
            return makeError(Error::InvalidArgument);
        }
        const uint64_t indexElementSize = clusterIndexByteSize(source.indexFormat);
        const uint64_t indexCount = uint64_t(source.triangleCount) * 3u;
        const uint64_t requiredIndexBytes = (indexCount - 1u) * source.indexBufferStride + indexElementSize;
        const uint64_t requiredVertexBytes = uint64_t(source.vertexCount - 1u) * source.vertexBufferStride + vertexInfo.bytesPerBlock;
        if (!indexElementSize || source.indexBufferStride < indexElementSize ||
            !validCommandSlice(*impl_, source.indexBuffer, BufferUsageBits::AccelerationStructureBuildInput, requiredIndexBytes, indexElementSize) ||
            !validCommandSlice(*impl_, source.vertexBuffer, BufferUsageBits::AccelerationStructureBuildInput, requiredVertexBytes) ||
            !validCommandSlice(*impl_, source.destinationBuffer, BufferUsageBits::AccelerationStructureStorage, 1, limits.clusterByteAlignment)) {
            return makeError(Error::InvalidArgument);
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
        buildInfo.indexBuffer = source.indexBuffer.deviceAddress();
        buildInfo.vertexBuffer = source.vertexBuffer.deviceAddress();
        destinationAddresses[index] = source.destinationBuffer.deviceAddress();

        totalTriangleCount += source.triangleCount;
        totalVertexCount += source.vertexCount;
        if (totalTriangleCount > std::numeric_limits<uint32_t>::max() ||
            totalVertexCount > std::numeric_limits<uint32_t>::max()) {
            return makeError(Error::InvalidArgument);
        }
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
        .maxAccelerationStructureCount = static_cast<uint32_t>(desc.clusters.size()),
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
        .opInput = {.pTriangleClusters = &triangleInput},
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    impl_->device->functions.vkGetClusterAccelerationStructureBuildSizesNV(impl_->device->device, &input, &sizes);
    auto scratch = alignedBuildScratch(*impl_, desc.scratchBuffer, limits.clusterScratchByteAlignment, sizes.buildScratchSize);
    if (!scratch) { return makeError(scratch.error()); }
    auto singleInput = input;
    auto singleTriangleInput = triangleInput;
    singleTriangleInput.maxTotalTriangleCount = desc.maxClusterTriangleCount;
    singleTriangleInput.maxTotalVertexCount = desc.maxClusterVertexCount;
    singleInput.maxAccelerationStructureCount = 1;
    singleInput.opInput.pTriangleClusters = &singleTriangleInput;
    impl_->device->functions.vkGetClusterAccelerationStructureBuildSizesNV(impl_->device->device, &singleInput, &sizes);
    for (const auto& cluster : desc.clusters) {
        if (cluster.destinationBuffer.size() < sizes.accelerationStructureSize) { return makeError(Error::InvalidArgument); }
    }
    auto result = uploadBufferSlice(desc.buildInfoBuffer, buildInfos.data(), buildInfoBytes);
    if (!result) { return result; }
    result = uploadBufferSlice(desc.destinationAddressBuffer, destinationAddresses.data(), destinationAddressBytes);
    if (!result) { return result; }
    result = retainBufferSlices(*this, {desc.scratchBuffer, desc.buildInfoBuffer, desc.destinationAddressBuffer, desc.destinationSizeBuffer});
    if (!result) { return result; }
    for (const auto& cluster : desc.clusters) {
        result = retainBufferSlices(*this, {cluster.indexBuffer, cluster.vertexBuffer, cluster.destinationBuffer});
        if (!result) { return result; }
    }
    const uint64_t buildInfoAddress = desc.buildInfoBuffer.deviceAddress();
    const uint64_t destinationAddress = desc.destinationAddressBuffer.deviceAddress();
    VkClusterAccelerationStructureCommandsInfoNV commands{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
        .input = input,
        .scratchData = scratch->deviceAddress(),
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

    if (desc.destinationSizeBuffer.valid()) {
        commands.dstSizesArray = {desc.destinationSizeBuffer.deviceAddress(), sizeof(uint32_t), uint64_t(desc.clusters.size()) * sizeof(uint32_t)};
    }
    if (std::getenv("METALLIC_TRACE_CLAS")) {
        spdlog::info("[CLAS Trace] build count={} infos={:x} destinations={:x} scratch={:x} sizes={:x} firstDst={:x}",
            desc.clusters.size(), buildInfoAddress, destinationAddress, commands.scratchData,
            commands.dstSizesArray.deviceAddress, destinationAddresses.front());
    }

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
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, inputDependency);
    impl_->device->functions.vkCmdBuildClusterAccelerationStructureIndirectNV(impl_->commandBuffer, &commands);

    const VkMemoryBarrier2 outputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : 0),
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_ACCESS_2_SHADER_READ_BIT : 0),
    };
    const VkDependencyInfo outputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &outputBarrier,
    };
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, outputDependency);
    return {};
#endif
}

Result<ClusterAccelerationStructureBuildSizes> Device::queryClusterAccelerationStructureMoveSizes(
    uint32_t maxCount,
    uint64_t maxBytes) const
{
    ClusterAccelerationStructureBuildSizes buildSizes{};
#ifndef VK_NV_cluster_acceleration_structure
    return makeError(Error::Unsupported);
#else
    if (!impl_ || !impl_->capabilities.clusterAccelerationStructure) { return makeError(Error::Unsupported); }
    if (!maxCount || !maxBytes) { return makeError(Error::InvalidArgument); }

    VkClusterAccelerationStructureMoveObjectsInputNV move{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV,
        .type = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV,
        .noMoveOverlap = VK_TRUE, .maxMovedBytes = maxBytes};
    VkClusterAccelerationStructureInputInfoNV input{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = maxCount,
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
        .opInput = {.pMoveObjects = &move}};
    VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    impl_->functions.vkGetClusterAccelerationStructureBuildSizesNV(impl_->device, &input, &sizes);
    buildSizes = {sizes.accelerationStructureSize, sizes.updateScratchSize, sizes.buildScratchSize};
    return buildSizes;
#endif
}

Result<> CommandBuffer::moveClusterAccelerationStructures(const ClusterAccelerationStructureMoveDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT)) {
        return makeError(Error::InvalidArgument);
    }
#ifndef VK_NV_cluster_acceleration_structure
    return makeError(Error::Unsupported);
#else
    if (!impl_->device->capabilities.clusterAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
    const uint64_t arrayBytes = uint64_t(desc.objects.size()) * sizeof(uint64_t);
    if (desc.objects.empty() || desc.objects.size() > UINT32_MAX ||
        !validCommandSlice(*impl_, desc.sourceAddressBuffer, BufferUsageBits::AccelerationStructureBuildInput, arrayBytes, 8) ||
        !validCommandSlice(*impl_, desc.destinationAddressBuffer, BufferUsageBits::AccelerationStructureStorage, arrayBytes, 8) ||
        detail::BufferAddressCommandAccess::overlap(desc.sourceAddressBuffer, desc.destinationAddressBuffer)) {
        return makeError(Error::InvalidArgument);
    }
    std::vector<uint64_t> sources(desc.objects.size()), destinations(desc.objects.size());
    uint64_t totalBytes = 0;
    const auto& limits = impl_->device->physicalProperties.cluster;
    for (uint32_t i = 0; i < desc.objects.size(); ++i) {
        const auto& item = desc.objects[i];
        if (!validCommandSlice(*impl_, item.sourceBuffer, BufferUsageBits::AccelerationStructureStorage, 1, limits.clusterByteAlignment) ||
            !validCommandSlice(*impl_, item.destinationBuffer, BufferUsageBits::AccelerationStructureStorage, item.sourceBuffer.size(), limits.clusterByteAlignment) ||
            totalBytes > UINT64_MAX - item.sourceBuffer.size()) {
            return makeError(Error::InvalidArgument);
        }
        sources[i] = item.sourceBuffer.deviceAddress();
        destinations[i] = item.destinationBuffer.deviceAddress();
        totalBytes += item.sourceBuffer.size();
    }
    struct MoveRange { uint64_t start, end; bool destination; };
    std::vector<MoveRange> ranges;
    ranges.reserve(size_t(desc.objects.size()) * 2);
    for (uint32_t i = 0; i < desc.objects.size(); ++i) {
        ranges.push_back({sources[i], sources[i] + desc.objects[i].sourceBuffer.size(), false});
        ranges.push_back({destinations[i], destinations[i] + desc.objects[i].sourceBuffer.size(), true});
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
    uint64_t sourceEnd = 0, destinationEnd = 0;
    for (const auto& range : ranges) {
        if (range.start < destinationEnd || (range.destination && range.start < sourceEnd)) {
            return makeError(Error::InvalidArgument);
        }
        if (range.destination) { destinationEnd = std::max(destinationEnd, range.end); }
        else { sourceEnd = std::max(sourceEnd, range.end); }
    }
    VkClusterAccelerationStructureMoveObjectsInputNV move{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV,
        .type = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV,
        .noMoveOverlap = VK_TRUE, .maxMovedBytes = totalBytes};
    VkClusterAccelerationStructureInputInfoNV input{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = static_cast<uint32_t>(desc.objects.size()),
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
        .opInput = {.pMoveObjects = &move}};
    VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    impl_->device->functions.vkGetClusterAccelerationStructureBuildSizesNV(impl_->device->device, &input, &sizes);
    // MOVE_OBJECTS reports its scratch requirement in updateScratchSize.
    auto scratch = alignedBuildScratch(*impl_, desc.scratchBuffer, limits.clusterScratchByteAlignment, sizes.updateScratchSize);
    if (!scratch) { return makeError(scratch.error()); }
    for (const auto& pair : {std::pair{desc.sourceAddressBuffer, sources.data()},
                             std::pair{desc.destinationAddressBuffer, destinations.data()}}) {
        auto result = uploadBufferSlice(pair.first, pair.second, arrayBytes);
        if (!result) { return result; }
    }
    auto retained = retainBufferSlices(*this, {desc.sourceAddressBuffer, desc.destinationAddressBuffer, desc.scratchBuffer});
    if (!retained) { return retained; }
    for (const auto& item : desc.objects) {
        retained = retainBufferSlices(*this, {item.sourceBuffer, item.destinationBuffer});
        if (!retained) { return retained; }
    }
    VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR};
    VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, dependency);
    VkClusterAccelerationStructureCommandsInfoNV commands{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
        .input = input, .scratchData = scratch->deviceAddress(),
        .dstAddressesArray = {desc.destinationAddressBuffer.deviceAddress(), sizeof(uint64_t), arrayBytes},
        .srcInfosArray = {desc.sourceAddressBuffer.deviceAddress(), sizeof(uint64_t), arrayBytes}};
    if (std::getenv("METALLIC_TRACE_CLAS")) {
        spdlog::info("[CLAS Trace] move count={} sources={:x} destinations={:x} scratch={:x} firstSrc={:x} firstDst={:x} bytes={} scratchCapacity={} scratchRequired={}",
            desc.objects.size(), commands.srcInfosArray.deviceAddress, commands.dstAddressesArray.deviceAddress,
            commands.scratchData, sources.front(), destinations.front(), totalBytes,
            scratch->size(), sizes.updateScratchSize);
    }
    impl_->device->functions.vkCmdBuildClusterAccelerationStructureIndirectNV(impl_->commandBuffer, &commands);
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, dependency);
    return {};
#endif
}

Result<> CommandBuffer::buildClusterAccelerationStructureBottomLevels(
    const ClusterAccelerationStructureBottomLevelBuildDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT)) {
        return makeError(Error::InvalidArgument);
    }
#ifndef VK_NV_cluster_acceleration_structure
    (void)desc;
    return makeError(Error::Unsupported);
#else
    if (!impl_->device->capabilities.clusterAccelerationStructure ||
        impl_->device->functions.vkCmdBuildClusterAccelerationStructureIndirectNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    if (!desc.maxClusterCountPerAccelerationStructure || !desc.maxTotalClusterCount || !desc.maxAccelerationStructureCount ||
        desc.buildInfoStride > UINT64_MAX / desc.maxAccelerationStructureCount ||
        desc.destinationAddressStride > UINT64_MAX / desc.maxAccelerationStructureCount ||
        desc.destinationSizeStride > UINT64_MAX / desc.maxAccelerationStructureCount ||
        desc.buildInfoStride < sizeof(ClusterAccelerationStructureBottomLevelBuildInfo) || desc.buildInfoStride % 8 ||
        desc.destinationAddressStride < sizeof(uint64_t) || desc.destinationAddressStride % 8 ||
        !validCommandSlice(*impl_, desc.buildInfoBuffer, BufferUsageBits::AccelerationStructureBuildInput,
            uint64_t(desc.maxAccelerationStructureCount) * desc.buildInfoStride, 8) ||
        !validCommandSlice(*impl_, desc.destinationAddressBuffer, BufferUsageBits::AccelerationStructureStorage,
            uint64_t(desc.maxAccelerationStructureCount) * desc.destinationAddressStride, 8) ||
        (desc.buildInfoCountBuffer.valid() && !validCommandSlice(*impl_, desc.buildInfoCountBuffer, BufferUsageBits::Storage, sizeof(uint32_t), 4)) ||
        (desc.destinationSizeBuffer.valid() && (desc.destinationSizeStride < sizeof(uint32_t) || desc.destinationSizeStride % 4 ||
            !validCommandSlice(*impl_, desc.destinationSizeBuffer, BufferUsageBits::Storage,
                uint64_t(desc.maxAccelerationStructureCount) * desc.destinationSizeStride, 4)))) {
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
    VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    impl_->device->functions.vkGetClusterAccelerationStructureBuildSizesNV(impl_->device->device, &input, &sizes);
    const auto& limits = impl_->device->physicalProperties.cluster;
    auto scratch = alignedBuildScratch(*impl_, desc.scratchBuffer, limits.clusterScratchByteAlignment, sizes.buildScratchSize);
    if (!scratch) { return makeError(scratch.error()); }
    if (desc.destinationMode == ClusterAccelerationStructureDestinationMode::Implicit &&
        !validCommandSlice(*impl_, desc.destinationStorageBuffer, BufferUsageBits::AccelerationStructureStorage,
            sizes.accelerationStructureSize, limits.clusterBottomLevelByteAlignment)) {
        return makeError(Error::InvalidArgument);
    }
    auto retained = retainBufferSlices(*this, {desc.buildInfoBuffer, desc.buildInfoCountBuffer, desc.destinationStorageBuffer,
        desc.destinationAddressBuffer, desc.destinationSizeBuffer, desc.scratchBuffer});
    if (!retained) { return retained; }
    VkClusterAccelerationStructureCommandsInfoNV commands{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
        .input = input,
        .dstImplicitData = desc.destinationMode == ClusterAccelerationStructureDestinationMode::Implicit
            ? desc.destinationStorageBuffer.deviceAddress() : 0,
        .scratchData = scratch->deviceAddress(),
        .dstAddressesArray = {desc.destinationAddressBuffer.deviceAddress(), desc.destinationAddressStride, desc.destinationAddressBuffer.size()},
        .dstSizesArray = {desc.destinationSizeBuffer.deviceAddress(), desc.destinationSizeStride, desc.destinationSizeBuffer.size()},
        .srcInfosArray = {desc.buildInfoBuffer.deviceAddress(), desc.buildInfoStride, desc.buildInfoBuffer.size()},
        .srcInfosCount = desc.buildInfoCountBuffer.deviceAddress(),
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
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, inputDependency);
    impl_->device->functions.vkCmdBuildClusterAccelerationStructureIndirectNV(impl_->commandBuffer, &commands);

    const VkMemoryBarrier2 outputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : 0),
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_ACCESS_2_SHADER_READ_BIT : 0),
    };
    const VkDependencyInfo outputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &outputBarrier,
    };
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, outputDependency);
    return {};
#endif
}

Result<> CommandBuffer::buildPartitionedAccelerationStructure(
    const PartitionedAccelerationStructureBuildDesc& desc)
{
    if (!validCommandRecording(impl_.get(), recording_, VK_QUEUE_COMPUTE_BIT)) {
        return makeError(Error::InvalidArgument);
    }
#ifndef VK_NV_partitioned_acceleration_structure
    (void)desc;
    return makeError(Error::Unsupported);
#else
    if (!impl_->device->capabilities.partitionedAccelerationStructure ||
        impl_->device->functions.vkCmdBuildPartitionedAccelerationStructuresNV == nullptr) {
        return makeError(Error::Unsupported);
    }
    if (desc.destination == nullptr || desc.destination->impl_ == nullptr ||
        !desc.destination->valid() ||
        desc.destination->desc().topLevelBackend != RayTracingTopLevelBackend::Partitioned ||
        !desc.destination->impl_->partitioned ||
        desc.destination->impl_->device != impl_->device ||
        desc.instanceCount == 0 ||
        desc.instanceCount != desc.destination->impl_->partitioned->desc.inputs.instanceCount ||
        !validCommandSlice(*impl_, desc.instanceBuffer, BufferUsageBits::AccelerationStructureBuildInput,
            uint64_t(desc.instanceCount) * sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV), 16) ||
        !validCommandSlice(*impl_, desc.scratchBuffer, BufferUsageBits::Storage)) {
        return makeError(Error::InvalidArgument);
    }
    const auto queueCanAccess = [&](const Buffer& buffer) {
        const auto families = detail::queueFamiliesForAccess(*impl_->device, buffer.desc().queueAccess);
        return std::find(families.begin(), families.end(), impl_->queueFamilyIndex) != families.end();
    };
    if (!queueCanAccess(*desc.destination->impl_->storage)) { return makeError(Error::InvalidArgument); }
    const uint64_t instanceAddress = desc.instanceBuffer.deviceAddress();
    auto scratch = alignedBuildScratch(*impl_, desc.scratchBuffer, 256,
        desc.destination->impl_->partitioned->desc.sizes.buildScratchSize);
    if (!scratch) { return makeError(scratch.error()); }
    const uint64_t scratchAddress = scratch->deviceAddress();

    VkBuildPartitionedAccelerationStructureIndirectCommandNV operation{
        .opType = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV,
        .argCount = desc.instanceCount,
        .argData = VkStridedDeviceAddressNV{
            .startAddress = instanceAddress,
            .strideInBytes = sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV),
        },
    };
    // The instance upload can rotate while an earlier build is still in flight.
    // Keep its indirect operation immutable through this recording's completion.
    auto operationUploadResult = Device::createBuffer(impl_->device, BufferDesc{
        .size = sizeof(operation) + sizeof(uint32_t),
        .usage = BufferUsageBits::Storage | BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        .memoryLocation = MemoryLocation::HostUpload,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
    });
    if (!operationUploadResult) { return makeError(operationUploadResult.error()); }
    auto operationUpload = std::move(*operationUploadResult);
    const uint64_t operationAddress = operationUpload->deviceAddress();
    const uint64_t operationCountAddress = operationAddress + sizeof(operation);
    if (!operationAddress) { return makeError(Error::Failure); }
    auto* mappedOperation = static_cast<uint8_t*>(operationUpload->map());
    if (!mappedOperation) { return makeError(Error::Failure); }
    std::memcpy(mappedOperation, &operation, sizeof(operation));
    const uint32_t operationCount = 1;
    std::memcpy(mappedOperation + sizeof(operation), &operationCount, sizeof(operationCount));
    operationUpload->flush();
    operationUpload->unmap();

    const PartitionedAccelerationStructureBuildInputs& inputs =
        desc.destination->impl_->partitioned->desc.inputs;
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
    auto retained = retainResource(desc.destination->retainAllocation());
    if (retained) { retained = retainResource(desc.instanceBuffer.retainAllocation()); }
    if (retained) { retained = retainResource(desc.scratchBuffer.retainAllocation()); }
    if (retained) { retained = retainResource(operationUpload->retainAllocation()); }
    if (!retained) { return retained; }
    const VkMemoryBarrier2 inputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT |
            (desc.graphManagedSynchronization ? 0 : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT),
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT |
            (desc.graphManagedSynchronization ? 0 : VK_ACCESS_2_MEMORY_WRITE_BIT),
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    };
    const VkDependencyInfo inputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &inputBarrier,
    };
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, inputDependency);
    impl_->device->functions.vkCmdBuildPartitionedAccelerationStructuresNV(impl_->commandBuffer, &buildInfo);

    if (desc.graphManagedSynchronization) { return {}; }

    const VkMemoryBarrier2 outputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : 0),
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            (impl_->device->capabilities.rayQuery ? VK_ACCESS_2_SHADER_READ_BIT : 0),
    };
    const VkDependencyInfo outputDependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &outputBarrier,
    };
    vulkan::recordBarrier(impl_->device->functions, impl_->device->device, impl_->commandBuffer, outputDependency);
    return {};
#endif
}

METALLIC_RHI_HANDLE_DEFINITIONS(CommandPool)

Result<> CommandPool::reset()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    const Result<> result = resultFromVk(impl_->device->functions.vkResetCommandPool(impl_->device->device, impl_->pool, 0));
    if (result) { impl_->submissions->cancel(); }
    return result;
}

Result<std::unique_ptr<CommandBuffer>> CommandPool::createCommandBuffer()
{
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
    if (impl_->recycleForCapture && !impl_->availableBuffers.empty()) {
        commandBuffer = impl_->availableBuffers.back();
        impl_->availableBuffers.pop_back();
    } else {
        const VkResult result = impl_->device->functions.vkAllocateCommandBuffers(impl_->device->device, &allocateInfo, &commandBuffer);
        if (result != VK_SUCCESS) {
            return std::unexpected(resultFromVk(result).error());
        }
    }

    auto commandBufferImpl = std::make_unique<detail::CommandBufferImpl>();
    commandBufferImpl->submissions = impl_->submissions;
    commandBufferImpl->device = impl_->device;
    commandBufferImpl->pool = impl_->pool;
    commandBufferImpl->commandBuffer = commandBuffer;
    commandBufferImpl->capturePool = impl_->recycleForCapture ? impl_.get() : nullptr;
    commandBufferImpl->queueFamilyIndex = impl_->queueFamilyIndex;
    commandBufferImpl->queueFlags = impl_->queueFlags;
    return std::unique_ptr<CommandBuffer>(new CommandBuffer(std::move(commandBufferImpl)));
}

METALLIC_RHI_HANDLE_DEFINITIONS(Swapchain)

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

DisplayOutputMode Swapchain::outputMode() const
{
    return impl_ != nullptr ? impl_->outputMode : DisplayOutputMode::SDR;
}

Texture* Swapchain::texture(uint32_t imageIndex)
{
    if (impl_ == nullptr || imageIndex >= impl_->textures.size()) {
        return nullptr;
    }
    return impl_->textures[imageIndex].get();
}

Result<uint32_t> Swapchain::acquireNextImage(SwapchainSemaphore& semaphore)
{
    uint32_t imageIndex{};
    if (impl_ == nullptr || semaphore.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    const VkResult result = impl_->device->functions.vkAcquireNextImageKHR(
        impl_->device->device,
        impl_->swapchain,
        kAcquireTimeoutNanoseconds,
        semaphore.impl_->semaphore,
        VK_NULL_HANDLE,
        &imageIndex);
    // SUBOPTIMAL still acquires an image and schedules a semaphore signal. The
    // caller must submit/present it, rather than reusing an unconsumed semaphore.
    if (result == VK_SUBOPTIMAL_KHR) {
        return imageIndex;
    }
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return makeError(Error::OutOfDate);
    }
    if (result != VK_SUCCESS) { return std::unexpected(resultFromVk(result).error()); }
    return imageIndex;
}

Result<> Swapchain::present(Queue& queue, uint32_t imageIndex, SwapchainSemaphore& waitSemaphore)
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

    if (vulkan::toolingHooks().requiresPresentDrain()) {
        const auto idle = vulkan::waitInterop(queue);
        if (idle != VK_SUCCESS) { return resultFromVk(idle); }
    }
    const VkResult result = vulkan::presentInterop(queue, presentInfo);
    if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
        return makeError(Error::OutOfDate);
    }
    return resultFromVk(result);
}

METALLIC_RHI_HANDLE_DEFINITIONS(Device)

const void* Device::identity() const
{
    return impl_.get();
}

Result<std::shared_ptr<void>> Device::sharedState(
    const void* key, const std::function<Result<std::shared_ptr<void>>()>& factory)
{
    if (!impl_ || !key || !factory) { return makeError(Error::InvalidArgument); }
    std::lock_guard lock(impl_->sharedStateMutex);
    const auto existing = impl_->sharedStates.find(key);
    if (existing != impl_->sharedStates.end()) { return existing->second; }
    auto state = factory();
    if (!state) { return makeError(state.error()); }
    if (!*state) { return makeError(Error::InvalidArgument); }
    impl_->sharedStates.emplace(key, *state);
    return *state;
}

MemoryBudgetReservation::~MemoryBudgetReservation() { reset(); }
MemoryBudgetReservation::MemoryBudgetReservation(MemoryBudgetReservation&& other) noexcept
    : state_(std::move(other.state_)), bytes_(std::exchange(other.bytes_, 0)) {}
MemoryBudgetReservation& MemoryBudgetReservation::operator=(MemoryBudgetReservation&& other) noexcept
{
    if (this != &other) { reset(); state_ = std::move(other.state_); bytes_ = std::exchange(other.bytes_, 0); }
    return *this;
}
void MemoryBudgetReservation::reset()
{
    if (state_ && bytes_) {
        std::lock_guard lock(state_->mutex);
        state_->reservedBytes -= bytes_;
    }
    bytes_ = 0;
    state_.reset();
}
DeviceMemoryBudget Device::memoryBudget() const
{
    if (!impl_ || !impl_->allocator) { return {}; }
    std::lock_guard lock(impl_->memoryBudgetState->mutex);
    return impl_->memoryBudgetLocked();
}
void Device::setMemoryBudgetPolicy(const MemoryBudgetPolicy& policy)
{
    if (!impl_) { return; }
    std::lock_guard lock(impl_->memoryBudgetState->mutex);
    impl_->memoryBudgetState->policy = policy;
}
Result<MemoryBudgetReservation> Device::reserveMemoryBudget(uint64_t bytes)
{
    MemoryBudgetReservation reservation{};
    if (!impl_ || !impl_->allocator) { return makeError(Error::InvalidArgument); }
    std::lock_guard lock(impl_->memoryBudgetState->mutex);
    auto& state = *impl_->memoryBudgetState;
    if (!state.policy.enabled || !bytes) { return reservation; }
    const auto budget = impl_->memoryBudgetLocked();
    if (bytes > budget.availableBytes) {
        ++state.deniedAllocations;
        spdlog::error("[GPUBudget] Reservation denied requested={} available={} existingReservations={}", bytes, budget.availableBytes, state.reservedBytes);
        return makeError(Error::OutOfMemory);
    }
    state.reservedBytes += bytes;
    reservation.state_ = impl_->memoryBudgetState;
    reservation.bytes_ = bytes;
    return reservation;
}
void Device::logMemoryBudget(const char* phase) const
{
    if (!impl_ || !impl_->allocator) { return; }
    DeviceMemoryBudget budget;
    {
        std::lock_guard lock(impl_->memoryBudgetState->mutex);
        if (!impl_->memoryBudgetState->policy.enabled) { return; }
        // Phase markers around external SDK calls must observe their new usage,
        // even when less than the normal refresh interval has elapsed.
        impl_->lastBudgetRefresh = {};
        budget = impl_->memoryBudgetLocked();
    }
    for (uint32_t i = 0; i < budget.heaps.size(); ++i) {
        const auto& h = budget.heaps[i];
        spdlog::info("[GPUBudget] {} heap={} local={} driverBudget={} usage={} budget={} blocks={}/{} allocations={}/{} reserved={} safety={} primaryAvailable={} denied={}",
            phase, i, h.deviceLocal, budget.driverBudget, h.usageBytes, h.budgetBytes, h.blockCount, h.blockBytes,
            h.allocationCount, h.allocationBytes, budget.reservedBytes, budget.policy.safetyBytes, budget.availableBytes, budget.deniedAllocations);
    }
    static constexpr const char* names[] = {"other", "geometry", "clas", "clasScratch", "rtas", "textures", "frame", "upload"};
    for (size_t i = 0; i < budget.domains.size(); ++i) {
        const auto& d = budget.domains[i];
        spdlog::info("[GPUBudget] {} domain={} liveAndRetained={} deviceLocal={} peak={} allocations={}",
            phase, names[i], d.allocationBytes, d.deviceLocalBytes, d.peakAllocationBytes, d.allocationCount);
    }
}

const DeviceCapabilities& Device::capabilities() const
{
    static const DeviceCapabilities emptyCapabilities;
    return impl_ != nullptr ? impl_->capabilities : emptyCapabilities;
}

Result<RayTracingAccelerationStructureProperties> Device::queryRayTracingAccelerationStructureProperties() const
{
    RayTracingAccelerationStructureProperties queriedProperties{};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    queriedProperties = RayTracingAccelerationStructureProperties{
        .scratchAlignment = rayTracingScratchAlignment(*impl_),
        .instanceBufferAlignment = 16,
        .instanceRecordSize = sizeof(RayTracingGPUInstance),
    };
    return queriedProperties;
}

Result<RayTracingAccelerationStructureBuildSizes> Device::queryRayTracingAccelerationStructureBuildSizes(
    const RayTracingAccelerationStructureBuildInputs& inputs) const
{
    if (!impl_) { return makeError(Error::InvalidArgument); }
    if (!impl_->capabilities.rayTracingAccelerationStructure) { return makeError(Error::Unsupported); }
    if (hasFlag(inputs.flags, RayTracingAccelerationStructureBuildFlags::AllowDataAccess) &&
        !impl_->capabilities.rayTracingPositionFetch) { return makeError(Error::Unsupported); }
    if (inputs.type == RayTracingAccelerationStructureType::BottomLevel) {
        if (inputs.instanceCount != 0 || std::any_of(inputs.geometries.begin(), inputs.geometries.end(),
                [](const auto& geometry) { return geometry.coverage != nullptr; })) {
            return makeError(Error::InvalidArgument);
        }
        return queryBottomLevelBuildSizes(*impl_, inputs.geometries, {}, inputs.flags);
    }
    if (inputs.type != RayTracingAccelerationStructureType::TopLevel ||
        !inputs.geometries.empty() || !inputs.instanceCount) { return makeError(Error::InvalidArgument); }
    VkAccelerationStructureGeometryKHR geometry{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR};
    geometry.geometry.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    const VkAccelerationStructureBuildGeometryInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = toVkAccelerationStructureBuildFlags(inputs.flags),
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1, .pGeometries = &geometry,
    };
    VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    impl_->functions.vkGetAccelerationStructureBuildSizesKHR(impl_->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &inputs.instanceCount, &sizes);
    if (!sizes.accelerationStructureSize || !sizes.buildScratchSize) { return makeError(Error::Failure); }
    return RayTracingAccelerationStructureBuildSizes{sizes.accelerationStructureSize, sizes.buildScratchSize, sizes.updateScratchSize};
}

Result<std::unique_ptr<RayTracingBottomLevelBuildPlan>> Device::prepareRayTracingBottomLevelBuild(
    std::span<const RayTracingTriangleGeometryDesc> geometries,
    RayTracingAccelerationStructureBuildFlags flags)
{
    if (!impl_ || geometries.empty() || geometries.size() > UINT32_MAX) { return makeError(Error::InvalidArgument); }
    if (!impl_->capabilities.rayTracingAccelerationStructure) { return makeError(Error::Unsupported); }
    if (hasFlag(flags, RayTracingAccelerationStructureBuildFlags::AllowDataAccess) &&
        !impl_->capabilities.rayTracingPositionFetch) { return makeError(Error::Unsupported); }
    for (const auto& geometry : geometries) {
        if (!validTriangleGeometry(impl_.get(), geometry)) { return makeError(Error::InvalidArgument); }
        if (geometry.coverage && (geometry.coverage->triangles.size() != geometry.primitiveCount ||
                hasFlag(geometry.flags, RayTracingGeometryFlags::Opaque))) { return makeError(Error::InvalidArgument); }
    }
    auto plan = std::make_unique<detail::RayTracingBottomLevelBuildPlanImpl>();
    plan->device = impl_.get();
    plan->flags = flags;
    plan->geometries.assign(geometries.begin(), geometries.end());
    plan->coverage.resize(geometries.size());
    plan->bakeBudget = impl_->coverageBakeBytes;
    constexpr uint64_t kMaxPreparedBakeBytes = 256ull * 1024 * 1024;
    const auto prepareCoverage = [&](const RayTracingCoverageDesc& coverage,
                                     detail::PreparedGeometryCoverage& prepared) -> Result<> {
        if (coverage.mode != RayTracingCoverageMode::Mask && coverage.mode != RayTracingCoverageMode::Blend) {
            return makeError(Error::InvalidArgument);
        }
        if (!impl_->vulkanCapabilities.opacityMicromap ||
            hasFlag(flags, RayTracingAccelerationStructureBuildFlags::AllowUpdate)) { return {}; }
        const auto& properties = impl_->physicalProperties.opacityMicromap;
        if (coverage.triangles.size() > properties.maxMicromapTriangles) { return {}; }
        if (!detail::OpacityMicromapBaker(coverage).bake(std::min(4u, properties.maxOpacity4StateSubdivisionLevel), prepared.baked)) { return {}; }
        if (prepared.baked.stateCounts[0] + prepared.baked.stateCounts[1] == 0) {
            prepared.baked = {};
            return {};
        }
        const uint64_t packedBytes = prepared.baked.data.size() + prepared.baked.triangles.size() * sizeof(OpacityMicromapTriangle);
        auto liveBytes = plan->bakeBudget->load(std::memory_order_relaxed);
        for (;;) {
            if (liveBytes > kMaxPreparedBakeBytes || packedBytes > kMaxPreparedBakeBytes - liveBytes) {
                prepared.baked = {};
                return {};
            }
            if (plan->bakeBudget->compare_exchange_weak(liveBytes, liveBytes + packedBytes, std::memory_order_relaxed)) { break; }
        }
        plan->bakeBytes += packedBytes;
        const auto fallback = [&]() {
            plan->bakeBudget->fetch_sub(packedBytes, std::memory_order_relaxed);
            plan->bakeBytes -= packedBytes;
            prepared = {};
        };
        const auto micromapFlags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace;
        const OpacityMicromapBuildInput input{.usages = prepared.baked.usages};
        std::vector<VkMicromapUsageKHR> usages;
        VkAccelerationStructureGeometryMicromapDataKHR data{};
        VkAccelerationStructureGeometryKHR geometry{};
        const auto valid = makeOpacityMicromapGeometry(properties, true, impl_->opacityMicromapExt,
            &input, micromapFlags, false, usages, data, geometry);
        if (!valid) { fallback(); return {}; }
        if (impl_->opacityMicromapExt) {
            std::vector<VkMicromapUsageEXT> extUsages;
            const auto info = makeExtMicromapBuildInfo(data, micromapFlags, extUsages);
            VkMicromapBuildSizesInfoEXT sizes{.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
            impl_->functions.vkGetMicromapBuildSizesEXT(impl_->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &sizes);
            prepared.sizes = {sizes.micromapSize, sizes.buildScratchSize, 0};
        } else {
            const VkAccelerationStructureBuildGeometryInfoKHR info{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                .type = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR,
                .flags = toVkAccelerationStructureBuildFlags(micromapFlags),
                .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                .geometryCount = 1, .pGeometries = &geometry,
            };
            VkAccelerationStructureBuildSizesInfoKHR sizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            impl_->functions.vkGetAccelerationStructureBuildSizesKHR(impl_->device,
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, nullptr, &sizes);
            prepared.sizes = {sizes.accelerationStructureSize, sizes.buildScratchSize, 0};
        }
        if (!prepared.sizes.accelerationStructureSize || prepared.sizes.accelerationStructureSize > UINT64_MAX - 255) {
            fallback(); return {};
        }
        auto storage = createBuffer({
            .size = prepared.sizes.accelerationStructureSize + 255,
            .usage = BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        });
        if (!storage) {
            const auto error = storage.error();
            fallback();
            return error == Error::DeviceLost ? Result<>(makeError(error)) : Result<>();
        }
        auto resource = std::make_shared<detail::RayTracingAccelerationStructureImpl>();
        resource->device = impl_.get();
        resource->desc = {.buildFlags = micromapFlags, .size = prepared.sizes.accelerationStructureSize};
        resource->storage = std::move(*storage);
        VkResult result;
        if (impl_->opacityMicromapExt) {
            const VkMicromapCreateInfoEXT info{
                .sType = VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT,
                .buffer = resource->storage->impl_->buffer,
                .offset = (0 - resource->storage->deviceAddress()) & 255,
                .size = prepared.sizes.accelerationStructureSize,
                .type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT,
            };
            result = impl_->functions.vkCreateMicromapEXT(impl_->device, &info, nullptr, &resource->micromap);
        } else {
            const auto create = reinterpret_cast<PFN_vkCreateAccelerationStructure2KHR>(
                impl_->instanceFunctions.vkGetDeviceProcAddr(impl_->device, "vkCreateAccelerationStructure2KHR"));
            if (!create) { fallback(); return {}; }
            const VkAccelerationStructureCreateInfo2KHR info{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_2_KHR,
                .addressRange = {.address = (resource->storage->deviceAddress() + 255) & ~uint64_t(255),
                    .size = prepared.sizes.accelerationStructureSize},
                .type = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR,
            };
            result = create(impl_->device, &info, nullptr, &resource->accelerationStructure);
        }
        if (result != VK_SUCCESS) {
            const auto error = resultFromVk(result).error();
            fallback();
            return error == Error::DeviceLost ? Result<>(makeError(error)) : Result<>();
        }
        prepared.resource = std::move(resource);
        return {};
    };
    for (size_t index = 0; index < geometries.size(); ++index) {
        plan->geometries[index].coverage = nullptr;
        if (!geometries[index].coverage) { continue; }
        const auto prepared = prepareCoverage(*geometries[index].coverage, plan->coverage[index]);
        if (!prepared) { return makeError(prepared.error()); }
        const auto& coverage = plan->coverage[index];
        if (!coverage.resource) { continue; }
        ++plan->coverageStats.geometryCount;
        plan->coverageStats.triangleCount += geometries[index].primitiveCount;
        plan->coverageStats.storageBytes += coverage.resource->storage->memoryInfo().sizeBytes;
    }
    auto sizes = queryBottomLevelBuildSizes(*impl_, plan->geometries, plan->coverage, flags);
    if (!sizes) { return makeError(sizes.error()); }
    plan->sizes = *sizes;
    for (const auto& coverage : plan->coverage) {
        plan->sizes.buildScratchSize = std::max(plan->sizes.buildScratchSize, coverage.sizes.buildScratchSize);
    }
    return std::unique_ptr<RayTracingBottomLevelBuildPlan>(new RayTracingBottomLevelBuildPlan(std::move(plan)));
}

Result<std::unique_ptr<RayTracingAccelerationStructure>> Device::createRayTracingAccelerationStructure(
    const RayTracingBottomLevelBuildPlan& plan)
{
    if (!plan.valid() || plan.impl_->device != impl_.get() || plan.impl_->recorded.load()) { return makeError(Error::InvalidArgument); }
    auto resource = createRayTracingAccelerationStructure(RayTracingAccelerationStructureDesc{
        .type = RayTracingAccelerationStructureType::BottomLevel,
        .buildFlags = plan.impl_->flags, .size = plan.impl_->sizes.accelerationStructureSize,
    });
    if (!resource) { return makeError(resource.error()); }
    (*resource)->impl_->coverageBuildIdentity = plan.impl_->identity;
    (*resource)->impl_->coverageStats = plan.impl_->coverageStats;
    for (const auto& coverage : plan.impl_->coverage) {
        if (coverage.resource) { (*resource)->impl_->coverageDependencies.push_back(coverage.resource); }
    }
    return resource;
}

Result<std::unique_ptr<RayTracingAccelerationStructure>> Device::createRayTracingAccelerationStructure(
    const RayTracingAccelerationStructureDesc& desc)
{
    if (!impl_ || !desc.size || desc.topLevelBackend != RayTracingTopLevelBackend::Standard ||
        (desc.type != RayTracingAccelerationStructureType::BottomLevel && desc.type != RayTracingAccelerationStructureType::TopLevel)) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->capabilities.rayTracingAccelerationStructure) { return makeError(Error::Unsupported); }
    if (hasFlag(desc.buildFlags, RayTracingAccelerationStructureBuildFlags::AllowDataAccess) &&
        !impl_->capabilities.rayTracingPositionFetch) { return makeError(Error::Unsupported); }
    auto storage = createBuffer({.size = desc.size,
        .usage = BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::ShaderDeviceAddress,
        .memoryLocation = MemoryLocation::Device, .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
    if (!storage) { return makeError(storage.error()); }
    const VkAccelerationStructureCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = (*storage)->impl_->buffer, .size = desc.size, .type = toVkAccelerationStructureType(desc.type)};
    auto resource = std::make_unique<detail::RayTracingAccelerationStructureImpl>();
    resource->device = impl_.get();
    resource->desc = desc;
    resource->storage = std::move(*storage);
    const VkResult result = impl_->functions.vkCreateAccelerationStructureKHR(impl_->device, &info, nullptr, &resource->accelerationStructure);
    if (result != VK_SUCCESS) { return makeError(resultFromVk(result).error()); }
    const VkAccelerationStructureDeviceAddressInfoKHR addressInfo{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = resource->accelerationStructure};
    resource->address = impl_->functions.vkGetAccelerationStructureDeviceAddressKHR(impl_->device, &addressInfo);
    if (!resource->address) { return makeError(Error::Failure); }
    return std::unique_ptr<RayTracingAccelerationStructure>(new RayTracingAccelerationStructure(std::move(resource)));
}

Result<std::unique_ptr<Buffer>> Device::createRayTracingInstanceBuffer(std::span<const RayTracingInstanceDesc> instances)
{
    std::unique_ptr<Buffer> buffer{};
    if (impl_ == nullptr || instances.size() > UINT32_MAX || instances.size() == 0) {
        return makeError(Error::InvalidArgument);
    }
    Result<> result = createBuffer(BufferDesc{
            .size = static_cast<uint64_t>(instances.size()) * sizeof(RayTracingGPUInstance),
            .structureStride = sizeof(RayTracingGPUInstance),
            .usage = BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        }).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
    if (!result) {
        return std::unexpected(result.error());
    }
    result = writeRayTracingInstances(*buffer, instances);
    if (!result) {
        return makeError(result.error());
    }
    return buffer;
}

Result<> Device::writeRayTracingInstances(
    Buffer& buffer,
    std::span<const RayTracingInstanceDesc> instances)
{
    if (impl_ == nullptr || buffer.impl_ == nullptr || buffer.impl_->device != impl_.get() ||
        instances.size() > UINT32_MAX || instances.size() == 0 ||
        !hasFlag(buffer.desc().usage, BufferUsageBits::AccelerationStructureBuildInput) ||
        static_cast<uint64_t>(instances.size()) * sizeof(RayTracingGPUInstance) >
            buffer.desc().size) {
        return makeError(Error::InvalidArgument);
    }

    static_assert(sizeof(RayTracingGPUInstance) == sizeof(VkAccelerationStructureInstanceKHR));
    static_assert(
        offsetof(RayTracingGPUInstance, accelerationStructureReference) ==
        offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference));
    std::vector<RayTracingGPUInstance> encoded(instances.size());
    for (uint32_t index = 0; index < instances.size(); ++index) {
        const RayTracingInstanceDesc& source = instances[index];
        if (source.bottomLevel == nullptr || source.bottomLevel->impl_ == nullptr ||
            source.bottomLevel->impl_->device != impl_.get() ||
            source.bottomLevel->impl_->desc.type !=
                RayTracingAccelerationStructureType::BottomLevel ||
            !source.bottomLevel->valid() || source.customIndex > 0x00ffffffu ||
            source.shaderBindingTableRecordOffset > 0x00ffffffu) {
            return makeError(Error::InvalidArgument);
        }
        RayTracingGPUInstance& destination = encoded[index];
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
    buffer.flush({0, byteSize});
    buffer.unmap();
    return {};
}

Result<ClusterAccelerationStructureProperties> Device::queryClusterAccelerationStructureProperties() const
{
    ClusterAccelerationStructureProperties queriedProperties{};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->capabilities.clusterAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
#ifndef VK_NV_cluster_acceleration_structure
    return makeError(Error::Unsupported);
#else
    const auto& properties = impl_->physicalProperties.cluster;
    if (properties.clusterByteAlignment == 0 ||
        properties.clusterScratchByteAlignment == 0) {
        return makeError(Error::Failure);
    }
    queriedProperties = ClusterAccelerationStructureProperties{
        .clusterStorageAlignment = properties.clusterByteAlignment,
        .bottomLevelStorageAlignment = properties.clusterBottomLevelByteAlignment,
        .scratchAlignment = properties.clusterScratchByteAlignment,
        .triangleBuildInfoSize =
            sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
        .bottomLevelBuildInfoSize =
            sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
    };
    return queriedProperties;
#endif
}

Result<ClusterAccelerationStructureBuildSizes> Device::queryClusterAccelerationStructureTriangleBuildSizes(
    const ClusterAccelerationStructureTriangleBuildSizesDesc& desc) const
{
    ClusterAccelerationStructureBuildSizes buildSizes{};
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->capabilities.clusterAccelerationStructure) {
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

    if (impl_->functions.vkGetClusterAccelerationStructureBuildSizesNV == nullptr) {
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
    impl_->functions.vkGetClusterAccelerationStructureBuildSizesNV(impl_->device, &inputInfo, &sizes);
    buildSizes = ClusterAccelerationStructureBuildSizes{
        .accelerationStructureSize = sizes.accelerationStructureSize,
        .updateScratchSize = sizes.updateScratchSize,
        .buildScratchSize = sizes.buildScratchSize,
    };
    return buildSizes;
#endif
}

Result<ClusterAccelerationStructureBuildSizes> Device::queryClusterAccelerationStructureBottomLevelBuildSizes(
    const ClusterAccelerationStructureBottomLevelBuildSizesDesc& desc) const
{
    ClusterAccelerationStructureBuildSizes buildSizes{};
    if (impl_ == nullptr ||
        desc.maxClusterCountPerAccelerationStructure == 0 ||
        desc.maxTotalClusterCount == 0 ||
        desc.maxAccelerationStructureCount == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->capabilities.clusterAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
#ifndef VK_NV_cluster_acceleration_structure
    return makeError(Error::Unsupported);
#else

    if (impl_->functions.vkGetClusterAccelerationStructureBuildSizesNV == nullptr) {
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
    impl_->functions.vkGetClusterAccelerationStructureBuildSizesNV(impl_->device, &inputInfo, &sizes);
    if (sizes.accelerationStructureSize == 0 || sizes.buildScratchSize == 0) {
        return makeError(Error::Failure);
    }
    buildSizes = ClusterAccelerationStructureBuildSizes{
        .accelerationStructureSize = sizes.accelerationStructureSize,
        .updateScratchSize = sizes.updateScratchSize,
        .buildScratchSize = sizes.buildScratchSize,
    };
    return buildSizes;
#endif
}

Result<PartitionedAccelerationStructureBuildSizes> Device::queryPartitionedAccelerationStructureBuildSizes(
    const PartitionedAccelerationStructureBuildInputs& inputs) const
{
    PartitionedAccelerationStructureBuildSizes buildSizes{};
    if (impl_ == nullptr || inputs.instanceCount == 0 ||
        inputs.partitionCount == 0 || inputs.maxInstancePerPartitionCount == 0 ||
        inputs.maxOperationCount == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->capabilities.partitionedAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
#ifndef VK_NV_partitioned_acceleration_structure
    return makeError(Error::Unsupported);
#else

    if (impl_->functions.vkGetPartitionedAccelerationStructuresBuildSizesNV == nullptr) {
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
    impl_->functions.vkGetPartitionedAccelerationStructuresBuildSizesNV(impl_->device, &inputInfo, &sizes);
    if (sizes.accelerationStructureSize == 0 || sizes.buildScratchSize == 0) {
        return makeError(Error::Failure);
    }
    buildSizes = PartitionedAccelerationStructureBuildSizes{
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
    return buildSizes;
#endif
}

Result<std::unique_ptr<RayTracingAccelerationStructure>> Device::createRayTracingAccelerationStructure(
    const PartitionedAccelerationStructureDesc& desc)
{
    if (impl_ == nullptr || desc.sizes.accelerationStructureSize == 0 ||
        desc.sizes.operationInfoSize == 0 || desc.sizes.operationCountSize == 0) {
        return makeError(Error::InvalidArgument);
    }
    PartitionedAccelerationStructureBuildSizes expectedSizes;
    Result<> result = queryPartitionedAccelerationStructureBuildSizes(desc.inputs).transform([&](auto rhiValue) { expectedSizes = std::move(rhiValue); });
    if (!result) {
        return std::unexpected(result.error());
    }
    if (desc.sizes.accelerationStructureSize < expectedSizes.accelerationStructureSize ||
        desc.sizes.operationInfoSize < expectedSizes.operationInfoSize ||
        desc.sizes.operationCountSize < expectedSizes.operationCountSize) {
        return makeError(Error::InvalidArgument);
    }

    auto implementation = std::make_unique<detail::RayTracingAccelerationStructureImpl>();
    implementation->device = impl_.get();
    implementation->desc = RayTracingAccelerationStructureDesc{
        .type = RayTracingAccelerationStructureType::TopLevel,
        .buildFlags = desc.inputs.flags,
        .size = desc.sizes.accelerationStructureSize,
        .topLevelBackend = RayTracingTopLevelBackend::Partitioned,
    };
    implementation->partitioned = std::make_unique<detail::PartitionedTopLevelState>();
    implementation->partitioned->desc = desc;
    result = createBuffer(BufferDesc{
            .size = desc.sizes.accelerationStructureSize,
            .usage = BufferUsageBits::AccelerationStructureStorage |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        }).transform([&](auto rhiValue) { implementation->storage = std::move(rhiValue); });
    if (!result) {
        return std::unexpected(result.error());
    }
    implementation->address = implementation->storage->deviceAddress();
    if (implementation->address == 0) {
        return makeError(Error::Failure);
    }
    return std::unique_ptr<RayTracingAccelerationStructure>(new RayTracingAccelerationStructure(std::move(implementation)));
}

Result<std::unique_ptr<Buffer>> Device::createPartitionedAccelerationStructureInstanceBuffer(std::span<const PartitionedAccelerationStructureInstanceDesc> instances)
{
    std::unique_ptr<Buffer> buffer{};
    if (impl_ == nullptr || instances.size() > UINT32_MAX || instances.size() == 0) {
        return makeError(Error::InvalidArgument);
    }
#ifndef VK_NV_partitioned_acceleration_structure
    return makeError(Error::Unsupported);
#else
    std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> encoded(instances.size());
    for (uint32_t index = 0; index < instances.size(); ++index) {
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
    Result<> result = createBuffer(BufferDesc{
            .size = static_cast<uint64_t>(encoded.size()) * sizeof(encoded[0]),
            .structureStride = sizeof(encoded[0]),
            .usage = BufferUsageBits::Storage |
                BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        }).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
    if (!result) {
        return std::unexpected(result.error());
    }
    void* mapped = buffer->map();
    if (mapped == nullptr) {
        buffer.reset();
        return makeError(Error::Failure);
    }
    const uint64_t byteSize = static_cast<uint64_t>(encoded.size()) * sizeof(encoded[0]);
    std::memcpy(mapped, encoded.data(), static_cast<size_t>(byteSize));
    buffer->flush({0, byteSize});
    buffer->unmap();
    return buffer;
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

Result<> Device::waitIdle()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    return resultFromVk(vulkan::waitInterop(*this));
}

Result<std::unique_ptr<Swapchain>> Device::createSwapchain(const SwapchainDesc& desc)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }


    auto swapchainImpl = std::make_unique<detail::SwapchainImpl>();
    swapchainImpl->device = impl_.get();
    const Result<> result = swapchainImpl->initialize(desc);
    if (!result) {
        return std::unexpected(result.error());
    }

    return std::unique_ptr<Swapchain>(new Swapchain(std::move(swapchainImpl)));
}

Result<std::unique_ptr<CommandPool>> Device::createCommandPool(Queue& queue)
{
    if (impl_ == nullptr ||
        queue.impl_ == nullptr ||
        queue.impl_->device != impl_.get()) {
        return makeError(Error::InvalidArgument);
    }


    auto poolImpl = std::make_unique<detail::CommandPoolImpl>();
    poolImpl->device = impl_.get();
    poolImpl->queueFamilyIndex = queue.impl_->familyIndex;
    poolImpl->queueFlags = queue.impl_->queueFlags;
    poolImpl->recycleForCapture = vulkan::toolingHooks().captureInjected();
    if (poolImpl->recycleForCapture) {
        std::lock_guard lock(impl_->captureCommandPoolMutex);
        auto& pools = impl_->captureCommandPools;
        const auto found = std::find_if(pools.begin(), pools.end(), [&](const auto& pool) {
            return pool.queueFamilyIndex == queue.impl_->familyIndex;
        });
        if (found != pools.end()) {
            poolImpl->pool = found->pool;
            poolImpl->availableBuffers = std::move(found->availableBuffers);
            pools.erase(found);
        }
    }
    if (poolImpl->pool != VK_NULL_HANDLE) {
        return std::unique_ptr<CommandPool>(new CommandPool(std::move(poolImpl)));
    }

    VkCommandPoolCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue.impl_->familyIndex,
    };

    VkCommandPool pool = VK_NULL_HANDLE;
    const VkResult result = impl_->functions.vkCreateCommandPool(impl_->device, &createInfo, nullptr, &pool);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    poolImpl->pool = pool;
    return std::unique_ptr<CommandPool>(new CommandPool(std::move(poolImpl)));
}

Result<std::unique_ptr<Fence>> Device::createFence(bool signaled)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }


    VkFenceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0u,
    };

    VkFence fence = VK_NULL_HANDLE;
    const VkResult result = impl_->functions.vkCreateFence(impl_->device, &createInfo, nullptr, &fence);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    auto fenceImpl = std::make_unique<detail::FenceImpl>();
    fenceImpl->device = impl_.get();
    fenceImpl->fence = fence;
    return std::unique_ptr<Fence>(new Fence(std::move(fenceImpl)));
}

Result<std::unique_ptr<TimestampQueryPool>> Device::createTimestampQueryPool(Queue& queue,
    const TimestampQueryPoolDesc& desc)
{
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

    VkQueryPoolCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = desc.queryCount,
    };
    VkQueryPool queryPool = VK_NULL_HANDLE;
    const VkResult result = impl_->functions.vkCreateQueryPool(impl_->device, &createInfo, nullptr, &queryPool);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    auto queryPoolImpl = std::make_unique<detail::TimestampQueryPoolImpl>();
    queryPoolImpl->device = impl_.get();
    queryPoolImpl->desc = desc;
    queryPoolImpl->queryPool = queryPool;
    queryPoolImpl->queueFamilyIndex = queue.impl_->familyIndex;
    queryPoolImpl->timestampValidBits = queue.impl_->timestampValidBits;
    queryPoolImpl->timestampPeriodNanoseconds = impl_->capabilities.timestampPeriodNanoseconds;
    return std::unique_ptr<TimestampQueryPool>(new TimestampQueryPool(std::move(queryPoolImpl)));
}

Result<std::unique_ptr<RayTracingAccelerationStructureCompactionQueryPool>> Device::createRayTracingAccelerationStructureCompactionQueryPool(
    const RayTracingAccelerationStructureCompactionQueryPoolDesc& desc)
{
    if (impl_ == nullptr || desc.queryCount == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }

    VkQueryPoolCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
        .queryCount = desc.queryCount,
    };
    VkQueryPool queryPool = VK_NULL_HANDLE;
    const VkResult result = impl_->functions.vkCreateQueryPool(impl_->device, &createInfo, nullptr, &queryPool);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    auto queryPoolImpl =
        std::make_unique<detail::RayTracingAccelerationStructureCompactionQueryPoolImpl>();
    queryPoolImpl->device = impl_.get();
    queryPoolImpl->desc = desc;
    queryPoolImpl->queryPool = queryPool;
    return std::unique_ptr<RayTracingAccelerationStructureCompactionQueryPool>(new RayTracingAccelerationStructureCompactionQueryPool(std::move(queryPoolImpl)));
}

Result<std::unique_ptr<Semaphore>> Device::createSemaphore()
{
    return createSemaphore(SemaphoreDesc{});
}

Result<std::unique_ptr<Semaphore>> Device::createSemaphore(const SemaphoreDesc& desc)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }


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
    const VkResult result = impl_->functions.vkCreateSemaphore(impl_->device, &createInfo, nullptr, &semaphore);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    auto semaphoreImpl = std::make_unique<detail::SemaphoreImpl>();
    semaphoreImpl->device = impl_.get();
    semaphoreImpl->semaphore = semaphore;
    return std::unique_ptr<Semaphore>(new Semaphore(std::move(semaphoreImpl)));
}

Result<std::unique_ptr<SwapchainSemaphore>> Device::createSwapchainSemaphore()
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }


    VkSemaphoreCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    VkSemaphore semaphore = VK_NULL_HANDLE;
    const VkResult result = impl_->functions.vkCreateSemaphore(impl_->device, &createInfo, nullptr, &semaphore);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    auto semaphoreImpl = std::make_unique<detail::SwapchainSemaphoreImpl>();
    semaphoreImpl->device = impl_.get();
    semaphoreImpl->semaphore = semaphore;
    return std::unique_ptr<SwapchainSemaphore>(new SwapchainSemaphore(std::move(semaphoreImpl)));
}

namespace {

BufferDesc effectiveBufferDesc(const BufferDesc& requestedDesc)
{
    auto desc = requestedDesc;
    // Diagnostic snapshots copy the bound storage allocations in both paths.
    const char* replay = std::getenv("METALLIC_WORK_CONTROL_REPLAY");
    if (replay && std::strcmp(replay, "1") == 0 && hasFlag(desc.usage, BufferUsageBits::Storage)) {
        desc.usage = desc.usage | BufferUsageBits::TransferSource;
    }
    return desc;
}

Result<VkBufferUsageFlags2> nativeBufferUsage(const detail::DeviceImpl& device, BufferUsageBits bufferUsage)
{
    const bool usesAccelerationStructure =
        hasFlag(bufferUsage, BufferUsageBits::AccelerationStructureBuildInput) ||
        hasFlag(bufferUsage, BufferUsageBits::AccelerationStructureStorage);
    if (usesAccelerationStructure && !device.capabilities.rayTracingAccelerationStructure) {
        return makeError(Error::Unsupported);
    }
    const bool requestsDeviceAddress = hasFlag(bufferUsage, BufferUsageBits::ShaderDeviceAddress) ||
        usesAccelerationStructure;
    if (requestsDeviceAddress && !device.bufferDeviceAddressEnabled) {
        return makeError(Error::Unsupported);
    }
    if (hasFlag(bufferUsage, BufferUsageBits::MemoryDecompression) && !device.capabilities.memoryDecompression) {
        return makeError(Error::Unsupported);
    }
    VkBufferUsageFlags2 usage = toVkBufferUsage(bufferUsage);
    if (device.opacityMicromapExt) {
        if (hasFlag(bufferUsage, BufferUsageBits::AccelerationStructureBuildInput)) {
            usage |= VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT;
        }
        if (hasFlag(bufferUsage, BufferUsageBits::AccelerationStructureStorage)) {
            usage |= VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT;
        }
    }
    if (device.bufferDeviceAddressEnabled &&
        (usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT)) != 0) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    return usage;
}

VkBufferCreateInfo nativeBufferInfo(const BufferDesc& desc,
    const VkBufferUsageFlags2CreateInfo& usage, const std::vector<uint32_t>& queueFamilies)
{
    return {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &usage,
        .size = desc.size,
        .sharingMode = queueFamilies.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = queueFamilies.size() > 1 ? uint32_t(queueFamilies.size()) : 0,
        .pQueueFamilyIndices = queueFamilies.size() > 1 ? queueFamilies.data() : nullptr,
    };
}

Result<> validateAliasBufferDesc(const detail::DeviceImpl& device, const BufferDesc& desc)
{
    constexpr auto allowedUsage = BufferUsageBits::Vertex | BufferUsageBits::Index | BufferUsageBits::Constant |
        BufferUsageBits::Storage | BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination |
        BufferUsageBits::ShaderDeviceAddress | BufferUsageBits::Indirect;
    constexpr auto allowedQueues = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy;
    if (!desc.size || desc.usage == BufferUsageBits::None || size_t(desc.memoryDomain) >= size_t(MemoryBudgetDomain::Count) ||
        (uint32_t(desc.queueAccess) & ~uint32_t(allowedQueues)) != 0) {
        return makeError(Error::InvalidArgument);
    }
    constexpr auto allUsage = allowedUsage | BufferUsageBits::AccelerationStructureBuildInput |
        BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::MemoryDecompression;
    if ((uint32_t(desc.usage) & ~uint32_t(allUsage)) != 0) { return makeError(Error::InvalidArgument); }
    // Acceleration-structure address graphs and decompression have additional
    // retention contracts. Scratch domains are excluded even for Storage usage.
    if (desc.memoryLocation != MemoryLocation::Device || (uint32_t(desc.usage) & ~uint32_t(allowedUsage)) != 0 ||
        (desc.memoryDomain != MemoryBudgetDomain::Other && desc.memoryDomain != MemoryBudgetDomain::FrameResources)) {
        return makeError(Error::Unsupported);
    }
    const auto& maintenance = device.physicalProperties.maintenance4;
    if (desc.size > maintenance.maxBufferSize) { return makeError(Error::InvalidArgument); }
    return {};
}

BufferAllocationRequirements aliasBufferRequirements(const VkMemoryRequirements& memory,
    const VkMemoryDedicatedRequirements& dedicated)
{
    return {.sizeBytes = memory.size, .alignmentBytes = memory.alignment, .memoryTypeBits = memory.memoryTypeBits,
        .requiresDedicatedAllocation = dedicated.requiresDedicatedAllocation != VK_FALSE};
}

} // namespace

Result<std::unique_ptr<Buffer>> Device::createBuffer(const BufferDesc& requestedDesc)
{
    return createBuffer(impl_.get(), requestedDesc);
}

Result<std::unique_ptr<Buffer>> Device::createBuffer(detail::DeviceImpl* impl_, const BufferDesc& requestedDesc)
{
    auto desc = effectiveBufferDesc(requestedDesc);
    if (impl_ == nullptr || desc.size == 0) {
        return makeError(Error::InvalidArgument);
    }


    const auto usageResult = nativeBufferUsage(*impl_, desc.usage);
    if (!usageResult) { return std::unexpected(usageResult.error()); }
    const auto usage = *usageResult;

    const std::vector<uint32_t> queueFamilies = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
    const VkBufferUsageFlags2CreateInfo usage2{
        .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
        .usage = usage,
    };
    const auto bufferInfo = nativeBufferInfo(desc, usage2, queueFamilies);

    VmaAllocationCreateInfo allocationInfo = allocationInfoForMemory(desc.memoryLocation);
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    if (size_t(desc.memoryDomain) >= size_t(MemoryBudgetDomain::Count)) { return makeError(Error::InvalidArgument); }
    const auto domain = desc.memoryDomain != MemoryBudgetDomain::Other ? desc.memoryDomain :
        desc.memoryLocation != MemoryLocation::Device ? MemoryBudgetDomain::Upload :
        hasFlag(desc.usage, BufferUsageBits::AccelerationStructureStorage) ? MemoryBudgetDomain::RayTracing : MemoryBudgetDomain::Other;
    std::unique_lock budgetLock(impl_->memoryBudgetState->mutex);
    // Pure staging does not benefit from occupying the device-local heap on a
    // discrete GPU with a large host-visible BAR. Keep directly GPU-read host
    // buffers on the normal AUTO policy, and let UMA fall back to its local heap.
    if (impl_->memoryBudgetState->policy.enabled && desc.memoryLocation == MemoryLocation::HostUpload &&
            desc.usage == BufferUsageBits::TransferSource) {
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }
    const Result<> admitted = impl_->prepareBufferAllocationLocked(bufferInfo, allocationInfo, domain);
    if (!admitted) { return std::unexpected(admitted.error()); }
    VmaAllocationInfo allocatedInfo{};
    const VkResult result = vmaCreateBuffer(
        impl_->allocator,
        &bufferInfo,
        &allocationInfo,
        &buffer,
        &allocation,
        &allocatedInfo);
    if (result != VK_SUCCESS) {
        spdlog::error(
            "[Vulkan] vmaCreateBuffer failed VkResult={} size={} usage=0x{:x} memoryLocation={} queueFamilyCount={}",
            static_cast<int32_t>(result),
            desc.size,
            static_cast<uint64_t>(usage),
            static_cast<uint32_t>(desc.memoryLocation),
            queueFamilies.size());
        return std::unexpected(resultFromVk(result).error());
    }

    auto bufferImpl = std::make_unique<detail::BufferImpl>();
    bufferImpl->device = impl_;
    bufferImpl->desc = desc;
    bufferImpl->desc.memoryDomain = domain;
    bufferImpl->buffer = buffer;
    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {
        const VkBufferDeviceAddressInfo addressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buffer};
        bufferImpl->address = impl_->functions.vkGetBufferDeviceAddress(impl_->device, &addressInfo);
    }
    bufferImpl->allocation = allocation;
    bufferImpl->memoryInfo = allocationMemoryInfo(allocatedInfo, impl_->memoryProperties,
        bufferImpl->memoryInfo.allocationId);
    bufferImpl->allocationBytes = allocatedInfo.size;
    bufferImpl->deviceLocal = (impl_->memoryProperties.memoryHeaps[impl_->memoryProperties.memoryTypes[allocatedInfo.memoryType].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    impl_->trackMemoryLocked(domain, allocatedInfo.size, bufferImpl->deviceLocal, true);
    return std::unique_ptr<Buffer>(new Buffer(std::move(bufferImpl)));
}

Result<uint64_t> Device::bufferAllocationSize(const BufferDesc& requestedDesc)
{
    const auto desc = effectiveBufferDesc(requestedDesc);
    constexpr auto allowedUsage = BufferUsageBits::Vertex | BufferUsageBits::Index | BufferUsageBits::Constant |
        BufferUsageBits::Storage | BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination |
        BufferUsageBits::ShaderDeviceAddress | BufferUsageBits::Indirect | BufferUsageBits::AccelerationStructureBuildInput |
        BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::MemoryDecompression;
    constexpr auto allowedQueues = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy;
    if (!impl_ || !desc.size || desc.usage == BufferUsageBits::None ||
        (uint32_t(desc.usage) & ~uint32_t(allowedUsage)) != 0 ||
        (uint32_t(desc.queueAccess) & ~uint32_t(allowedQueues)) != 0 ||
        size_t(desc.memoryDomain) >= size_t(MemoryBudgetDomain::Count)) {
        return makeError(Error::InvalidArgument);
    }

    const auto& maintenance = impl_->physicalProperties.maintenance4;
    if (desc.size > maintenance.maxBufferSize) { return makeError(Error::InvalidArgument); }
    const auto usage = nativeBufferUsage(*impl_, desc.usage);
    if (!usage) { return std::unexpected(usage.error()); }
    const auto families = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
    const VkBufferUsageFlags2CreateInfo usage2{
        .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO, .usage = *usage};
    const auto info = nativeBufferInfo(desc, usage2, families);
    VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    const VkDeviceBufferMemoryRequirements request{
        .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS, .pCreateInfo = &info};
    impl_->functions.vkGetDeviceBufferMemoryRequirements(impl_->device, &request, &requirements);
    if (!requirements.memoryRequirements.size) { return makeError(Error::Unsupported); }
    return requirements.memoryRequirements.size;
}

Result<BufferAllocationRequirements> Device::bufferAliasAllocationRequirements(const BufferDesc& requestedDesc)
{
    if (!impl_) { return makeError(Error::InvalidArgument); }
    const auto desc = effectiveBufferDesc(requestedDesc);
    const auto valid = validateAliasBufferDesc(*impl_, desc);
    if (!valid) { return std::unexpected(valid.error()); }

    const auto usage = nativeBufferUsage(*impl_, desc.usage);
    if (!usage) { return std::unexpected(usage.error()); }
    const auto families = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
    const VkBufferUsageFlags2CreateInfo usage2{
        .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO, .usage = *usage};
    const auto info = nativeBufferInfo(desc, usage2, families);
    VkMemoryDedicatedRequirements dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
    const VkDeviceBufferMemoryRequirements request{
        .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS, .pCreateInfo = &info};
    impl_->functions.vkGetDeviceBufferMemoryRequirements(impl_->device, &request, &requirements);
    if (!requirements.memoryRequirements.size || !requirements.memoryRequirements.alignment ||
        !requirements.memoryRequirements.memoryTypeBits) { return makeError(Error::Unsupported); }
    return aliasBufferRequirements(requirements.memoryRequirements, dedicated);
}

Result<std::vector<std::unique_ptr<Buffer>>> Device::createAliasedBuffers(std::span<const BufferDesc> descriptions)
{
    if (!impl_ || descriptions.empty()) { return makeError(Error::InvalidArgument); }

    const auto domain = descriptions.front().memoryDomain;
    std::vector<std::unique_ptr<detail::BufferImpl>> buffers;
    std::vector<BufferAllocationRequirements> bufferRequirements;
    buffers.reserve(descriptions.size());
    bufferRequirements.reserve(descriptions.size());
    VkMemoryRequirements combined{.size = 0, .alignment = 1, .memoryTypeBits = UINT32_MAX};
    std::vector<VkBufferUsageFlags2> bufferUsages;
    bufferUsages.reserve(descriptions.size());
    for (const auto& requestedDesc : descriptions) {
        const auto desc = effectiveBufferDesc(requestedDesc);
        const auto valid = validateAliasBufferDesc(*impl_, desc);
        if (!valid) { return std::unexpected(valid.error()); }
        if (desc.memoryDomain != domain) { return makeError(Error::InvalidArgument); }
        const auto usage = nativeBufferUsage(*impl_, desc.usage);
        if (!usage) { return std::unexpected(usage.error()); }
        const auto families = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
        const VkBufferUsageFlags2CreateInfo usage2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO, .usage = *usage};
        const auto info = nativeBufferInfo(desc, usage2, families);
        auto buffer = std::make_unique<detail::BufferImpl>();
        buffer->device = impl_.get();
        buffer->desc = desc;
        const VkResult created = impl_->functions.vkCreateBuffer(impl_->device, &info, nullptr, &buffer->buffer);
        if (created != VK_SUCCESS) { return std::unexpected(resultFromVk(created).error()); }
        VkMemoryDedicatedRequirements dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
        const VkBufferMemoryRequirementsInfo2 request{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2, .buffer = buffer->buffer};
        impl_->functions.vkGetBufferMemoryRequirements2(impl_->device, &request, &requirements);
        const auto& memory = requirements.memoryRequirements;
        if (dedicated.requiresDedicatedAllocation || !memory.size || !memory.alignment || !memory.memoryTypeBits) {
            return makeError(Error::Unsupported);
        }
        combined.size = std::max(combined.size, memory.size);
        combined.alignment = std::max(combined.alignment, memory.alignment);
        combined.memoryTypeBits &= memory.memoryTypeBits;
        bufferRequirements.push_back(aliasBufferRequirements(memory, dedicated));
        bufferUsages.push_back(*usage);
        buffers.push_back(std::move(buffer));
    }
    if (!combined.memoryTypeBits) { return makeError(Error::Unsupported); }
    auto backing = std::make_shared<detail::AliasBufferAllocation>();
    backing->device = impl_.get();
    backing->memoryDomain = domain;
    VmaAllocationInfo allocated{};
    {
        std::unique_lock budgetLock(impl_->memoryBudgetState->mutex);
        VmaAllocationCreateInfo allocationInfo{
            .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT,
            .usage = VMA_MEMORY_USAGE_UNKNOWN,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .memoryTypeBits = combined.memoryTypeBits,
        };
        uint32_t memoryType = 0;
        VkResult result = vmaFindMemoryTypeIndex(impl_->allocator, combined.memoryTypeBits, &allocationInfo, &memoryType);
        if (result != VK_SUCCESS) { return std::unexpected(resultFromVk(result).error()); }
        if (!impl_->admitMemoryLocked(memoryType, combined.size, domain)) { return makeError(Error::OutOfMemory); }
        allocationInfo.memoryTypeBits = 1u << memoryType;
        if (impl_->memoryBudgetState->policy.enabled) { allocationInfo.flags |= VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT; }
        VmaAllocation allocation = VK_NULL_HANDLE;
        // The allocator's BUFFER_DEVICE_ADDRESS flag also applies to raw memory
        // allocations. CAN_ALIAS avoids dedicating the backing to one VkBuffer.
        result = vmaAllocateMemory(impl_->allocator, &combined, &allocationInfo, &allocation, &allocated);
        if (result != VK_SUCCESS) { return std::unexpected(resultFromVk(result).error()); }
        backing->allocation = allocation;
        backing->sizeBytes = allocated.size;
        backing->deviceLocal = (impl_->memoryProperties.memoryHeaps[
            impl_->memoryProperties.memoryTypes[allocated.memoryType].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        impl_->trackMemoryLocked(domain, backing->sizeBytes, backing->deviceLocal, true);
    }
    // Retain on every member before binding so rollback destroys all VkBuffers
    // before the final owner frees the shared VkDeviceMemory.
    for (auto& buffer : buffers) { buffer->aliasAllocation = backing; }
    for (size_t index = 0; index < buffers.size(); ++index) {
        auto& buffer = *buffers[index];
        const auto& requirements = bufferRequirements[index];
        if (allocated.size < requirements.sizeBytes || allocated.offset % requirements.alignmentBytes ||
            !(requirements.memoryTypeBits & (1u << allocated.memoryType))) { return makeError(Error::Unsupported); }
        const VkResult bound = vmaBindBufferMemory(impl_->allocator, backing->allocation, buffer.buffer);
        if (bound != VK_SUCCESS) { return std::unexpected(resultFromVk(bound).error()); }
        if ((bufferUsages[index] & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {
            const VkBufferDeviceAddressInfo addressInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buffer.buffer};
            buffer.address = impl_->functions.vkGetBufferDeviceAddress(impl_->device, &addressInfo);
            if (!buffer.address) { return makeError(Error::Failure); }
        }
        buffer.memoryInfo = allocationMemoryInfo(allocated, impl_->memoryProperties, buffer.memoryInfo.allocationId);
        buffer.memoryInfo.backingAllocationId = backing->allocationId;
        buffer.memoryInfo.sizeBytes = requirements.sizeBytes;
        buffer.allocationBytes = requirements.sizeBytes;
        buffer.deviceLocal = backing->deviceLocal;
    }
    std::vector<std::unique_ptr<Buffer>> result;
    result.reserve(buffers.size());
    for (auto& buffer : buffers) { result.push_back(std::unique_ptr<Buffer>(new Buffer(std::move(buffer)))); }
    return result;
}

Result<std::unique_ptr<BufferView>> Device::createBufferView(Buffer& buffer,
    const BufferViewDesc& desc)
{
    if (impl_ == nullptr || buffer.impl_ == nullptr || buffer.impl_->device != impl_.get()) {
        return makeError(Error::InvalidArgument);
    }

    if (!impl_->capabilities.bindlessDescriptorHeap) {
        return makeError(Error::Unsupported);
    }

    const auto range = desc.range.resolve(buffer.impl_->desc.size);
    if (!range || range->size == 0) {
        return makeError(Error::InvalidArgument);
    }

    const uint64_t viewSize = range->size;
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

    const VkDeviceAddress bufferAddress = buffer.impl_->address;
    if (bufferAddress == 0) {
        return makeError(Error::Failure);
    }

    auto viewImpl = std::make_unique<detail::BufferViewImpl>();
    viewImpl->device = impl_.get();
    viewImpl->buffer = buffer.impl_;
    viewImpl->desc = desc;
    viewImpl->desc.range.size = viewSize;
    viewImpl->desc.structureStride = structureStride;
    viewImpl->descriptorType = descriptorType;
    viewImpl->address = bufferAddress + desc.range.offset;
    viewImpl->size = viewSize;
    return std::unique_ptr<BufferView>(new BufferView(std::move(viewImpl)));
}

namespace {
Result<VkImageCreateInfo> aliasTextureImageInfo(detail::DeviceImpl& device, const TextureDesc& desc,
    const std::vector<uint32_t>& queueFamilies)
{
    if (!desc.width || !desc.height || desc.depth != 1 || !desc.mipCount || !desc.layerCount ||
        toVkFormat(desc.format) == VK_FORMAT_UNDEFINED || size_t(desc.memoryDomain) >= size_t(MemoryBudgetDomain::Count)) {
        return makeError(Error::InvalidArgument);
    }
    constexpr auto allowedUsage = TextureUsageBits::Sampled | TextureUsageBits::Storage |
        TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource | TextureUsageBits::TransferDestination;
    constexpr auto allowedQueues = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy;
    if (desc.usage == TextureUsageBits::None || (uint32_t(desc.usage) & ~uint32_t(allowedUsage)) != 0 ||
        (uint32_t(desc.queueAccess) & ~uint32_t(allowedQueues)) != 0) {
        return makeError(Error::InvalidArgument);
    }
    if (desc.type != TextureType::Texture2D || desc.memoryLocation != MemoryLocation::Device ||
        desc.format == Format::D32Sfloat ||
        (desc.memoryDomain != MemoryBudgetDomain::Other && desc.memoryDomain != MemoryBudgetDomain::FrameResources)) {
        return makeError(Error::Unsupported);
    }
    uint32_t maxMipCount = 0;
    for (uint32_t extent = std::max(desc.width, desc.height); extent; extent >>= 1) { ++maxMipCount; }
    if (desc.mipCount > maxMipCount) { return makeError(Error::InvalidArgument); }
    VkImageCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = VK_IMAGE_CREATE_ALIAS_BIT,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = toVkFormat(desc.format),
        .extent = {desc.width, desc.height, 1},
        .mipLevels = desc.mipCount,
        .arrayLayers = desc.layerCount,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = toVkImageUsage(desc.usage),
        .sharingMode = queueFamilies.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = queueFamilies.size() > 1 ? uint32_t(queueFamilies.size()) : 0,
        .pQueueFamilyIndices = queueFamilies.size() > 1 ? queueFamilies.data() : nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImageFormatProperties properties{};
    const VkResult supported = device.instanceFunctions.vkGetPhysicalDeviceImageFormatProperties(device.physicalDevice, info.format,
        info.imageType, info.tiling, info.usage, info.flags, &properties);
    if (supported == VK_ERROR_FORMAT_NOT_SUPPORTED) { return makeError(Error::Unsupported); }
    if (supported != VK_SUCCESS) { return std::unexpected(resultFromVk(supported).error()); }
    if (desc.width > properties.maxExtent.width || desc.height > properties.maxExtent.height ||
        desc.mipCount > properties.maxMipLevels || desc.layerCount > properties.maxArrayLayers ||
        !(properties.sampleCounts & VK_SAMPLE_COUNT_1_BIT)) {
        return makeError(Error::InvalidArgument);
    }
    return info;
}

TextureAllocationRequirements aliasTextureRequirements(const VkMemoryRequirements& memory,
    const VkMemoryDedicatedRequirements& dedicated)
{
    return {.sizeBytes = memory.size, .alignmentBytes = memory.alignment,
        .memoryTypeBits = memory.memoryTypeBits,
        .requiresDedicatedAllocation = dedicated.requiresDedicatedAllocation != VK_FALSE,
        .prefersDedicatedAllocation = dedicated.prefersDedicatedAllocation != VK_FALSE};
}
} // namespace

Result<TextureAllocationRequirements> Device::textureAliasAllocationRequirements(const TextureDesc& desc)
{
    if (!impl_) { return makeError(Error::InvalidArgument); }

    const auto families = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
    const auto info = aliasTextureImageInfo(*impl_, desc, families);
    if (!info) { return std::unexpected(info.error()); }
    VkMemoryDedicatedRequirements dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
    const VkDeviceImageMemoryRequirements request{
        .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &*info};
    impl_->functions.vkGetDeviceImageMemoryRequirements(impl_->device, &request, &requirements);
    if (!requirements.memoryRequirements.size || !requirements.memoryRequirements.alignment ||
        !requirements.memoryRequirements.memoryTypeBits) { return makeError(Error::Unsupported); }
    return aliasTextureRequirements(requirements.memoryRequirements, dedicated);
}

Result<std::vector<std::unique_ptr<Texture>>> Device::createAliasedTextures(std::span<const TextureDesc> descriptions)
{
    if (!impl_ || descriptions.empty()) { return makeError(Error::InvalidArgument); }

    std::vector<std::unique_ptr<detail::TextureImpl>> images;
    std::vector<TextureAllocationRequirements> imageRequirements;
    images.reserve(descriptions.size());
    imageRequirements.reserve(descriptions.size());
    VkMemoryRequirements combined{.size = 0, .alignment = 1, .memoryTypeBits = UINT32_MAX};
    for (const auto& desc : descriptions) {
        const auto families = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
        const auto info = aliasTextureImageInfo(*impl_, desc, families);
        if (!info) { return std::unexpected(info.error()); }
        auto image = std::make_unique<detail::TextureImpl>();
        image->device = impl_.get();
        image->desc = desc;
        image->desc.memoryDomain = MemoryBudgetDomain::FrameResources;
        image->ownsImage = true;
        const VkResult created = impl_->functions.vkCreateImage(impl_->device, &*info, nullptr, &image->image);
        if (created != VK_SUCCESS) { return std::unexpected(resultFromVk(created).error()); }
        image->flags = info->flags;
        image->usage = info->usage;
        VkMemoryDedicatedRequirements dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
        const VkImageMemoryRequirementsInfo2 request{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = image->image};
        impl_->functions.vkGetImageMemoryRequirements2(impl_->device, &request, &requirements);
        const auto& memory = requirements.memoryRequirements;
        if (dedicated.requiresDedicatedAllocation || !memory.size || !memory.alignment || !memory.memoryTypeBits) {
            return makeError(Error::Unsupported);
        }
        combined.size = std::max(combined.size, memory.size);
        combined.alignment = std::max(combined.alignment, memory.alignment);
        combined.memoryTypeBits &= memory.memoryTypeBits;
        imageRequirements.push_back(aliasTextureRequirements(memory, dedicated));
        images.push_back(std::move(image));
    }
    if (!combined.memoryTypeBits) { return makeError(Error::Unsupported); }
    auto backing = std::make_shared<detail::AliasTextureAllocation>();
    backing->device = impl_.get();
    VmaAllocationInfo allocated{};
    {
        std::unique_lock budgetLock(impl_->memoryBudgetState->mutex);
        VmaAllocationCreateInfo allocationInfo{
            .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT,
            .usage = VMA_MEMORY_USAGE_UNKNOWN,
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .memoryTypeBits = combined.memoryTypeBits,
        };
        uint32_t memoryType = 0;
        VkResult result = vmaFindMemoryTypeIndex(impl_->allocator, combined.memoryTypeBits, &allocationInfo, &memoryType);
        if (result != VK_SUCCESS) { return std::unexpected(resultFromVk(result).error()); }
        if (!impl_->admitMemoryLocked(memoryType, combined.size, MemoryBudgetDomain::FrameResources)) {
            return makeError(Error::OutOfMemory);
        }
        allocationInfo.memoryTypeBits = 1u << memoryType;
        if (impl_->memoryBudgetState->policy.enabled) { allocationInfo.flags |= VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT; }
        VmaAllocation allocation = VK_NULL_HANDLE;
        result = vmaAllocateMemory(impl_->allocator, &combined, &allocationInfo, &allocation, &allocated);
        if (result != VK_SUCCESS) { return std::unexpected(resultFromVk(result).error()); }
        backing->allocation = allocation;
        backing->sizeBytes = allocated.size;
        backing->deviceLocal = (impl_->memoryProperties.memoryHeaps[
            impl_->memoryProperties.memoryTypes[allocated.memoryType].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        impl_->trackMemoryLocked(MemoryBudgetDomain::FrameResources, backing->sizeBytes, backing->deviceLocal, true);
    }
    // Retain the backing even for unbound members so rollback destroys every
    // native image before releasing the shared allocation.
    for (auto& image : images) { image->aliasAllocation = backing; }
    for (size_t index = 0; index < images.size(); ++index) {
        auto& image = *images[index];
        const auto& requirements = imageRequirements[index];
        if (allocated.size < requirements.sizeBytes || allocated.offset % requirements.alignmentBytes ||
            !(requirements.memoryTypeBits & (1u << allocated.memoryType))) { return makeError(Error::Unsupported); }
        const VkResult bound = vmaBindImageMemory(impl_->allocator, backing->allocation, image.image);
        if (bound != VK_SUCCESS) { return std::unexpected(resultFromVk(bound).error()); }
        image.memory = allocated.deviceMemory;
        image.memoryInfo = allocationMemoryInfo(allocated, impl_->memoryProperties, image.memoryInfo.allocationId);
        image.memoryInfo.backingAllocationId = backing->allocationId;
        image.memoryInfo.sizeBytes = requirements.sizeBytes;
        image.allocationSize = requirements.sizeBytes;
        image.deviceLocal = backing->deviceLocal;
    }
    std::vector<std::unique_ptr<Texture>> textures;
    textures.reserve(images.size());
    for (auto& image : images) { textures.push_back(std::unique_ptr<Texture>(new Texture(std::move(image)))); }
    return textures;
}

Result<uint64_t> Device::textureAllocationSize(const TextureDesc& desc)
{
    uint64_t allocationSize{};
    if (!impl_ || desc.format == Format::Unknown || !desc.width || !desc.height || !desc.mipCount) {
        return makeError(Error::InvalidArgument);
    }

    const auto families = detail::queueFamiliesForAccess(*impl_, desc.queueAccess);
    const VkImageCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = toVkImageType(desc.type), .format = toVkFormat(desc.format),
        .extent = {desc.width, desc.height, desc.depth}, .mipLevels = desc.mipCount,
        .arrayLayers = desc.layerCount, .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = toVkImageUsage(desc.usage),
        .sharingMode = families.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = families.size() > 1 ? uint32_t(families.size()) : 0,
        .pQueueFamilyIndices = families.size() > 1 ? families.data() : nullptr,
    };
    VkImage image = VK_NULL_HANDLE;
    const auto result = impl_->functions.vkCreateImage(impl_->device, &info, nullptr, &image);
    if (result != VK_SUCCESS) { return std::unexpected(resultFromVk(result).error()); }
    VkMemoryRequirements requirements{};
    impl_->functions.vkGetImageMemoryRequirements(impl_->device, image, &requirements);
    impl_->functions.vkDestroyImage(impl_->device, image, nullptr);
    allocationSize = requirements.size;
    return allocationSize;
}

Result<std::unique_ptr<Texture>> Device::createTexture(const TextureDesc& desc)
{
    if (impl_ == nullptr || desc.format == Format::Unknown) {
        return makeError(Error::InvalidArgument);
    }


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
    if (size_t(desc.memoryDomain) >= size_t(MemoryBudgetDomain::Count)) { return makeError(Error::InvalidArgument); }
    const auto domain = desc.memoryDomain != MemoryBudgetDomain::Other ? desc.memoryDomain :
        MemoryBudgetDomain::FrameResources;
    std::unique_lock budgetLock(impl_->memoryBudgetState->mutex);
    uint32_t memoryType = 0;
    VkResult typeResult = vmaFindMemoryTypeIndexForImageInfo(impl_->allocator, &imageInfo, &allocationInfo, &memoryType);
    if (typeResult != VK_SUCCESS) { return std::unexpected(resultFromVk(typeResult).error()); }
    VkMemoryDedicatedRequirements dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
    const VkDeviceImageMemoryRequirements request{.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &imageInfo};
    impl_->functions.vkGetDeviceImageMemoryRequirements(impl_->device, &request, &requirements);
    uint64_t heapGrowth = requirements.memoryRequirements.size;
    constexpr uint64_t kMaterialBlockBytes = 16ull * 1024 * 1024;
    if (domain == MemoryBudgetDomain::MaterialTextures && desc.memoryLocation == MemoryLocation::Device &&
            !dedicated.requiresDedicatedAllocation && heapGrowth <= kMaterialBlockBytes) {
        auto& pool = impl_->materialImagePools[memoryType];
        if (!pool) {
            const VmaPoolCreateInfo poolInfo{.memoryTypeIndex = memoryType, .blockSize = kMaterialBlockBytes};
            const VkResult created = vmaCreatePool(impl_->allocator, &poolInfo, &pool);
            if (created != VK_SUCCESS) { return std::unexpected(resultFromVk(created).error()); }
        }
        VmaDetailedStatistics stats{};
        vmaCalculatePoolStatistics(impl_->allocator, pool, &stats);
        // Existing free ranges already contribute to heapUsage. Otherwise admit
        // an entire new block, not just the small image's memory requirement.
        heapGrowth = stats.unusedRangeSizeMax >= requirements.memoryRequirements.size + requirements.memoryRequirements.alignment
            ? 0 : kMaterialBlockBytes;
        allocationInfo.pool = pool;
        allocationInfo.flags &= ~VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }
    if (!impl_->admitMemoryLocked(memoryType, heapGrowth, domain)) { return makeError(Error::OutOfMemory); }
    allocationInfo.memoryTypeBits = 1u << memoryType;
    if (impl_->memoryBudgetState->policy.enabled) { allocationInfo.flags |= VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT; }
    const VkResult result = vmaCreateImage(
        impl_->allocator,
        &imageInfo,
        &allocationInfo,
        &image,
        &allocation,
        &allocatedInfo);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    auto textureImpl = std::make_unique<detail::TextureImpl>();
    textureImpl->device = impl_.get();
    textureImpl->desc = desc;
    textureImpl->desc.memoryDomain = domain;
    textureImpl->image = image;
    textureImpl->memory = allocatedInfo.deviceMemory;
    textureImpl->allocation = allocation;
    textureImpl->memoryInfo = allocationMemoryInfo(allocatedInfo, impl_->memoryProperties,
        textureImpl->memoryInfo.allocationId);
    textureImpl->flags = imageInfo.flags;
    textureImpl->usage = imageInfo.usage;
    textureImpl->ownsImage = true;
    textureImpl->allocationSize = allocatedInfo.size;
    textureImpl->deviceLocal = (impl_->memoryProperties.memoryHeaps[impl_->memoryProperties.memoryTypes[allocatedInfo.memoryType].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    impl_->trackMemoryLocked(domain, allocatedInfo.size, textureImpl->deviceLocal, true);
    return std::unique_ptr<Texture>(new Texture(std::move(textureImpl)));
}

Result<std::unique_ptr<TextureView>> Device::createTextureView(Texture& texture,
    const TextureViewDesc& desc)
{
    if (impl_ == nullptr || texture.impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (texture.impl_->device != impl_.get()) { return makeError(Error::InvalidArgument); }
    const TextureDesc& textureDesc = texture.impl_->desc;
    const Format format = desc.format != Format::Unknown ? desc.format : textureDesc.format;
    // Images currently have no mutable-format flag. Validate semantic views now,
    // even if a native object is never needed.
    if (format != textureDesc.format || !desc.range.valid(textureDesc.mipCount, textureDesc.layerCount) ||
        (textureDesc.type == TextureType::Texture3D && (desc.range.baseLayer != 0 || desc.range.layerCount != 1))) {
        return makeError(Error::InvalidArgument);
    }
    for (auto component : desc.swizzle) {
        if (component > TextureViewDesc::Component::A) { return makeError(Error::InvalidArgument); }
    }

    auto viewImpl = std::make_unique<detail::TextureViewImpl>();
    viewImpl->device = impl_.get();
    viewImpl->texture = texture.impl_;
    viewImpl->desc = desc;
    viewImpl->desc.format = format;
    viewImpl->format = toVkFormat(format);
    return std::unique_ptr<TextureView>(new TextureView(std::move(viewImpl)));
}

Result<std::unique_ptr<ShaderModule>> Device::createShaderModule(const ShaderModuleDesc& desc)
{
    if (impl_ == nullptr || desc.spirv.size() < 5 || desc.spirv[0] != 0x07230203u) {
        return makeError(Error::InvalidArgument);
    }
    const uint64_t wordCount = desc.spirv.size_bytes() / sizeof(uint32_t);
    for (uint64_t offset = 5; offset < wordCount;) {
        const uint32_t count = desc.spirv.data()[offset] >> 16;
        if (count == 0 || count > wordCount - offset) { return makeError(Error::InvalidArgument); }
        // OpCapability DescriptorHeapEXT requires the native untyped-pointer path.
        if ((desc.spirv.data()[offset] & 0xffffu) == 17 && count == 2 && desc.spirv.data()[offset + 1] == 5128 &&
            (!impl_->bindlessDescriptorHeapEnabled || !impl_->shaderUntypedPointersEnabled)) {
            return makeError(Error::Unsupported);
        }
        offset += count;
    }


    std::string strideDiagnostics;
    if (!vulkan::validateNativeDescriptorHeapStrides(desc.spirv,
            {static_cast<uint32_t>(impl_->descriptorHeapWriter.resourceDescriptorStride()),
             static_cast<uint32_t>(impl_->descriptorHeapWriter.samplerDescriptorSize())}, strideDiagnostics)) {
        spdlog::error("Shader '{}' rejected: {}", desc.debugName ? desc.debugName : "<unnamed>", strideDiagnostics);
        return makeError(Error::InvalidArgument);
    }
    std::vector<uint32_t> opacityCode;
    ShaderModuleDesc deviceDesc = desc;
    if (impl_->vulkanCapabilities.opacityMicromap) {
        if (!vulkan::enableOpacityMicromapSpirv(
                deviceDesc.spirv, opacityCode, impl_->opacityMicromapExt)) {
            return makeError(Error::InvalidArgument);
        }
        deviceDesc.spirv = opacityCode;
    }
    VkShaderModuleCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = deviceDesc.spirv.size_bytes(),
        .pCode = deviceDesc.spirv.data(),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    const VkResult result = impl_->functions.vkCreateShaderModule(impl_->device, &createInfo, nullptr, &module);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }
    vulkan::toolingHooks().shaderBinary(deviceDesc.spirv.data(), deviceDesc.spirv.size_bytes());
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
    shaderImpl->hasDescriptorBindings = spirvHasDescriptorBindings(deviceDesc.spirv.data(), deviceDesc.spirv.size_bytes());
    shaderImpl->module = module;
    shaderImpl->deviceSpirv.assign(deviceDesc.spirv.begin(), deviceDesc.spirv.end());
    shaderImpl->contentHash = detail::shaderContentHash(deviceDesc);
    shaderImpl->inputSpirvFnv1a64 = detail::hashBytes(detail::kFnvOffset,
        desc.spirv.data(), desc.spirv.size_bytes());
    const char* replayCode = std::getenv("METALLIC_WORK_CONTROL_REPLAY");
    if (replayCode && std::strcmp(replayCode, "1") == 0) {
        const auto* input = reinterpret_cast<const uint8_t*>(desc.spirv.data());
        const auto* actual = reinterpret_cast<const uint8_t*>(deviceDesc.spirv.data());
        shaderImpl->replayInputSpirv.assign(input, input + desc.spirv.size_bytes());
        shaderImpl->replayDeviceSpirv.assign(actual, actual + deviceDesc.spirv.size_bytes());
    }
    if (impl_->pipelineExecutableStatistics) {
        shaderImpl->deviceSpirvFnv1a64 = detail::hashBytes(detail::kFnvOffset,
            deviceDesc.spirv.data(), deviceDesc.spirv.size_bytes());
        shaderImpl->diagnosticName = desc.debugName ? desc.debugName : "";
    }
    return std::unique_ptr<ShaderModule>(new ShaderModule(std::move(shaderImpl)));
}

Result<std::unique_ptr<PipelineCache>> Device::createPipelineCache(const PipelineCacheDesc& desc)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }


    auto cacheImpl = std::make_unique<detail::PipelineCacheImpl>();
    Result<> result = cacheImpl->initialize(*impl_, desc);
    if (!result) {
        return std::unexpected(result.error());
    }
    return std::unique_ptr<PipelineCache>(new PipelineCache(std::move(cacheImpl)));
}

bool Device::validShaderStage(const ShaderStageDesc& stage) const
{
    return impl_ && stage.module && stage.module->impl_ && stage.module->impl_->device == impl_.get() &&
        stage.entryPoint && stage.entryPoint[0] != '\0';
}

namespace {

enum class IndirectBinding { Pipeline, ShaderObject };

bool supportsIndirectBinding(const detail::DeviceImpl& device, VkShaderStageFlags stages, IndirectBinding binding)
{
    const auto& properties = device.physicalProperties.generatedCommands;
    const VkShaderStageFlags supported = binding == IndirectBinding::Pipeline
        ? properties.supportedIndirectCommandsShaderStagesPipelineBinding
        : properties.supportedIndirectCommandsShaderStagesShaderBinding;
    return device.capabilities.deviceGeneratedCommands && (supported & stages) == stages;
}

std::array<VkDescriptorSetAndBindingMappingEXT, 3> defaultHeapMappings(const DescriptorHeapWriter& writer)
{
    std::array<VkDescriptorSetAndBindingMappingEXT, 3> bindlessMappings{};
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
        static_cast<uint32_t>(writer.samplerDescriptorSize()));
    bindlessMappings[1] = makeHeapMapping(
        2,
        VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT |
            VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT |
            VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT,
        static_cast<uint32_t>(writer.resourceDescriptorStride()));
    bindlessMappings[2] = makeHeapMapping(
        2,
        VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT |
            VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
            VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT,
        static_cast<uint32_t>(writer.resourceDescriptorStride()));
    return bindlessMappings;
}

// The returned lock stays alive through Vulkan creation and recordPsoLocked().
Result<std::unique_lock<std::mutex>> lockPipelineCache(
    const detail::DeviceImpl& device, bool requested, detail::PipelineCacheImpl* cache)
{
    if (!requested) { return std::unique_lock<std::mutex>{}; }
    if (cache == nullptr || cache->device != &device || cache->pipelineCache == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }
    return std::unique_lock<std::mutex>(cache->mutex);
}

template<class PipelineImpl>
Result<std::unique_ptr<PipelineImpl>> createPipelineOwner(detail::DeviceImpl& device, bool usesBindlessHeap)
{
    auto pipeline = std::make_unique<PipelineImpl>();
    pipeline->device = &device;
    pipeline->usesBindlessHeap = usesBindlessHeap;
    if (!usesBindlessHeap) {
        const VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        const VkResult result = device.functions.vkCreatePipelineLayout(
            device.device, &layoutInfo, nullptr, &pipeline->layout);
        if (result != VK_SUCCESS) { return std::unexpected(resultFromVk(result).error()); }
    }
    // The Impl owns layout and pipeline throughout creation, including every early return.
    return pipeline;
}

} // namespace

Result<std::unique_ptr<GraphicsPipeline>> Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc)
{
    const bool usesTaskShader = desc.taskShader.module != nullptr;
    const bool usesMeshShader = desc.meshShader.module != nullptr;
    const bool usesVertexShader = desc.vertexShader.module != nullptr;
    if (impl_ == nullptr ||
        usesMeshShader == usesVertexShader ||
        (usesTaskShader && !usesMeshShader) ||
        !validShaderStage(desc.fragmentShader)) {
        return makeError(Error::InvalidArgument);
    }
    if (usesVertexShader && !validShaderStage(desc.vertexShader)) {
        return makeError(Error::InvalidArgument);
    }
    if (usesTaskShader && !validShaderStage(desc.taskShader)) {
        return makeError(Error::InvalidArgument);
    }
    if (usesMeshShader && !validShaderStage(desc.meshShader)) {
        return makeError(Error::InvalidArgument);
    }

    if (desc.indirectBindable && !supportsIndirectBinding(*impl_,
        (usesMeshShader ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT) | VK_SHADER_STAGE_FRAGMENT_BIT | (usesTaskShader ? VK_SHADER_STAGE_TASK_BIT_EXT : 0), IndirectBinding::Pipeline)) {
        return makeError(Error::Unsupported);
    }
    if (usesMeshShader && !impl_->capabilities.meshShader) {
        return makeError(Error::Unsupported);
    }
    if (usesTaskShader && !impl_->capabilities.taskShader) {
        return makeError(Error::Unsupported);
    }
    const bool configuresTaskSubgroups =
        desc.taskRequiredSubgroupSize != 0 ||
        desc.taskRequireFullSubgroups;
    if (configuresTaskSubgroups && !usesTaskShader) {
        return makeError(Error::InvalidArgument);
    }
    if (desc.taskRequireFullSubgroups &&
        desc.taskRequiredSubgroupSize == 0) {
        return makeError(Error::InvalidArgument);
    }
    if (desc.taskRequiredSubgroupSize != 0 &&
        (desc.taskRequiredSubgroupSize &
            (desc.taskRequiredSubgroupSize - 1u)) != 0) {
        return makeError(Error::InvalidArgument);
    }
    if (desc.taskRequiredSubgroupSize != 0 &&
        (!impl_->capabilities.subgroupSizeControl ||
         !impl_->capabilities.taskShaderSubgroupSizeControl ||
         desc.taskRequiredSubgroupSize <
             impl_->capabilities.minSubgroupSize ||
         desc.taskRequiredSubgroupSize >
             impl_->capabilities.maxSubgroupSize)) {
        return makeError(Error::Unsupported);
    }
    if (desc.taskRequireFullSubgroups &&
        !impl_->capabilities.computeFullSubgroups) {
        return makeError(Error::Unsupported);
    }
    if (desc.usesBindlessHeap && !impl_->capabilities.bindlessDescriptorHeap) {
        return makeError(Error::Unsupported);
    }
    const uint32_t colorCount = desc.colorAttachmentCount;
    if (colorCount > desc.colorFormats.size()) { return makeError(Error::InvalidArgument); }
    for (uint32_t index = 0; index < colorCount; ++index) {
        if (toVkFormat(desc.colorFormats[index]) == VK_FORMAT_UNDEFINED ||
            aspectForFormat(desc.colorFormats[index]) != VK_IMAGE_ASPECT_COLOR_BIT) {
            return makeError(Error::InvalidArgument);
        }
    }
    if (colorCount > impl_->physicalProperties.core.limits.maxColorAttachments) { return makeError(Error::Unsupported); }
    const bool hasColorFormat = colorCount != 0;
    const bool hasDepthStencilFormat = desc.depthStencilFormat != Format::Unknown;
    if (!hasColorFormat && !hasDepthStencilFormat) {
        return makeError(Error::InvalidArgument);
    }
    if ((desc.depthStencil.depthTestEnable || desc.depthStencil.depthWriteEnable) && !hasDepthStencilFormat) {
        return makeError(Error::InvalidArgument);
    }

    detail::PipelineCacheImpl* pipelineCache =
        desc.pipelineCache != nullptr ? desc.pipelineCache->impl_.get() : nullptr;
    auto pipelineCacheLock = lockPipelineCache(*impl_, desc.pipelineCache != nullptr, pipelineCache);
    if (!pipelineCacheLock) { return std::unexpected(pipelineCacheLock.error()); }
    const uint64_t psoHash = detail::graphicsPipelineStateHash(desc);

    const char* vertexEntryPoint = desc.vertexShader.entryPoint;
    const char* taskEntryPoint = desc.taskShader.entryPoint;
    const char* meshEntryPoint = desc.meshShader.entryPoint;
    const char* fragmentEntryPoint = desc.fragmentShader.entryPoint;
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(usesTaskShader ? 3u : 2u);
    size_t taskStageIndex = std::numeric_limits<size_t>::max();
    VkShaderStageFlags graphicsShaderStages = VK_SHADER_STAGE_FRAGMENT_BIT;
    if (usesMeshShader) {
#ifdef VK_EXT_mesh_shader
        if (usesTaskShader) {
            taskStageIndex = stages.size();
            stages.push_back(VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
                .module = desc.taskShader.module->impl_->module,
                .pName = taskEntryPoint,
            });
            graphicsShaderStages |= VK_SHADER_STAGE_TASK_BIT_EXT;
        }
        stages.push_back(VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
            .module = desc.meshShader.module->impl_->module,
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
            .module = desc.vertexShader.module->impl_->module,
            .pName = vertexEntryPoint,
        });
        graphicsShaderStages |= VK_SHADER_STAGE_VERTEX_BIT;
    }
    stages.push_back(VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = desc.fragmentShader.module->impl_->module,
        .pName = fragmentEntryPoint,
    });
    std::array<VkDescriptorSetAndBindingMappingEXT, 3> bindlessMappings{};
    VkShaderDescriptorSetAndBindingMappingInfoEXT bindlessMappingInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
    };
    if (desc.usesBindlessHeap) {
        bindlessMappings = defaultHeapMappings(impl_->descriptorHeapWriter);
        bindlessMappingInfo.mappingCount = static_cast<uint32_t>(bindlessMappings.size());
        bindlessMappingInfo.pMappings = bindlessMappings.data();
        for (VkPipelineShaderStageCreateInfo& stage : stages) {
            for (const ShaderModule* shader : {desc.vertexShader.module, desc.taskShader.module, desc.meshShader.module, desc.fragmentShader.module}) {
                if (shader != nullptr && shader->impl_->module == stage.module && shader->impl_->hasDescriptorBindings) {
                    stage.pNext = &bindlessMappingInfo;
                }
            }
        }
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo taskSubgroupSizeInfo{
        .sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = desc.taskRequiredSubgroupSize,
    };
    if (configuresTaskSubgroups) {
        if (taskStageIndex >= stages.size()) {
            return makeError(Error::InvalidArgument);
        }
        VkPipelineShaderStageCreateInfo& taskStage = stages[taskStageIndex];
        if (desc.taskRequireFullSubgroups) {
            taskStage.flags |=
                VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }
        if (desc.taskRequiredSubgroupSize != 0) {
            taskSubgroupSizeInfo.pNext = taskStage.pNext;
            taskStage.pNext = &taskSubgroupSizeInfo;
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
    std::array<VkPipelineColorBlendAttachmentState, GraphicsPipelineDesc::kMaxColorAttachments> colorBlendAttachments;
    colorBlendAttachments.fill(colorBlendAttachment);
    VkPipelineColorBlendStateCreateInfo colorBlendState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = colorCount,
        .pAttachments = colorBlendAttachments.data(),
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

    auto owner = createPipelineOwner<detail::GraphicsPipelineImpl>(*impl_, desc.usesBindlessHeap);
    if (!owner) { return std::unexpected(owner.error()); }
    auto pipelineImpl = std::move(*owner);
    const VkPipelineLayout layout = pipelineImpl->layout;
    VkResult result = VK_SUCCESS;

    std::array<VkFormat, GraphicsPipelineDesc::kMaxColorAttachments> colorFormats{};
    for (uint32_t index = 0; index < colorCount; ++index) { colorFormats[index] = toVkFormat(desc.colorFormats[index]); }
    const VkFormat depthStencilFormat = toVkFormat(desc.depthStencilFormat);
    VkPipelineRenderingCreateInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = colorCount,
        .pColorAttachmentFormats = colorFormats.data(),
        .depthAttachmentFormat = depthStencilFormat,
    };
    VkPipelineCreateFlags2CreateInfo bindlessPipelineFlags{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .pNext = &renderingInfo,
        .flags = (desc.usesBindlessHeap ? VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT : 0) |
            (desc.indirectBindable ? VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT : 0),
    };
    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = (desc.usesBindlessHeap || desc.indirectBindable) ? static_cast<const void*>(&bindlessPipelineFlags) :
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

    VkPipeline& pipeline = pipelineImpl->pipeline;
    result = impl_->functions.vkCreateGraphicsPipelines(
        impl_->device,
        pipelineCache != nullptr ? pipelineCache->pipelineCache : VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &pipeline);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    pipelineImpl->psoHash = psoHash;
    pipelineImpl->pipelineCacheHit = pipelineCache != nullptr &&
        pipelineCache->recordPsoLocked(psoHash);
    return std::unique_ptr<GraphicsPipeline>(new GraphicsPipeline(std::move(pipelineImpl)));
}

Result<std::unique_ptr<ComputePipeline>> Device::createComputePipeline(const ComputePipelineDesc& desc)
{
    return createComputePipelineImpl(desc, {});
}

Result<std::unique_ptr<ComputePipeline>> Device::createComputePipelineImpl(const ComputePipelineDesc& desc,
    std::span<const detail::ShaderBindingMappingDesc> mappings)
{
    if (impl_ == nullptr ||
        !validShaderStage(desc.computeShader) ||
        (mappings.size() > UINT32_MAX) ||
        (!desc.usesBindlessHeap && mappings.size() > 0)) {
        return makeError(Error::InvalidArgument);
    }

    if (desc.indirectBindable && !supportsIndirectBinding(*impl_,
        VK_SHADER_STAGE_COMPUTE_BIT, IndirectBinding::Pipeline)) {
        return makeError(Error::Unsupported);
    }
    if (desc.usesBindlessHeap) {
        if (!impl_->capabilities.bindlessDescriptorHeap) {
            return makeError(Error::Unsupported);
        }
        const VkDeviceSize requiredPushDataSize =
            desc.bindlessUserPushDataSize;
        if (impl_->descriptorHeapWriter.maxPushDataSize() < requiredPushDataSize) {
            return makeError(Error::Unsupported);
        }
    }

    detail::PipelineCacheImpl* pipelineCache =
        desc.pipelineCache != nullptr ? desc.pipelineCache->impl_.get() : nullptr;
    auto pipelineCacheLock = lockPipelineCache(*impl_, desc.pipelineCache != nullptr, pipelineCache);
    if (!pipelineCacheLock) { return std::unexpected(pipelineCacheLock.error()); }
    const uint64_t psoHash = detail::mappedComputePipelineStateHash(desc, mappings);

    const char* computeEntryPoint = desc.computeShader.entryPoint;
    VkPipelineShaderStageCreateInfo stage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = desc.computeShader.module->impl_->module,
        .pName = computeEntryPoint,
    };

    std::vector<VkDescriptorSetAndBindingMappingEXT> bindlessMappings;
    VkShaderDescriptorSetAndBindingMappingInfoEXT bindlessMappingInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
    };
    if (desc.usesBindlessHeap && desc.computeShader.module->impl_->hasDescriptorBindings) {
        if (mappings.size() == 0) {
            const auto defaults = defaultHeapMappings(impl_->descriptorHeapWriter);
            bindlessMappings.assign(defaults.begin(), defaults.end());
        } else {
            bindlessMappings.reserve(mappings.size());
            for (uint32_t index = 0; index < mappings.size(); ++index) {
                const detail::ShaderBindingMappingDesc& source = mappings[index];
                if (source.bindingCount == 0) {
                    return makeError(Error::InvalidArgument);
                }

                VkSpirvResourceTypeFlagsEXT resourceMask = 0;
                uint32_t descriptorStride = 0;
                switch (source.type) {
                case detail::ShaderBindingType::Sampler:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.samplerDescriptorSize());
                    break;
                case detail::ShaderBindingType::SampledImage:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.resourceDescriptorStride());
                    break;
                case detail::ShaderBindingType::StorageImage:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_ONLY_IMAGE_BIT_EXT |
                        VK_SPIRV_RESOURCE_TYPE_READ_WRITE_IMAGE_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.resourceDescriptorStride());
                    break;
                case detail::ShaderBindingType::ConstantBuffer:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.resourceDescriptorStride());
                    break;
                case detail::ShaderBindingType::StorageBuffer:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT |
                        VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.resourceDescriptorStride());
                    break;
                case detail::ShaderBindingType::AccelerationStructure:
                    resourceMask = VK_SPIRV_RESOURCE_TYPE_ACCELERATION_STRUCTURE_BIT_EXT;
                    descriptorStride = static_cast<uint32_t>(impl_->descriptorHeapWriter.resourceDescriptorStride());
                    break;
                }

                const uint32_t valueSize = source.source == detail::ShaderBindingSource::HeapConstantOffset
                    ? 0u
                    : source.source == detail::ShaderBindingSource::DeviceAddressFromPushData
                        ? static_cast<uint32_t>(sizeof(uint64_t))
                        : static_cast<uint32_t>(sizeof(uint32_t));
                if ((valueSize != 0 &&
                     (source.pushDataOffset > desc.bindlessUserPushDataSize ||
                      valueSize > desc.bindlessUserPushDataSize - source.pushDataOffset)) ||
                    (source.source == detail::ShaderBindingSource::DeviceAddressFromPushData &&
                     source.type != detail::ShaderBindingType::ConstantBuffer &&
                     source.type != detail::ShaderBindingType::StorageBuffer &&
                     source.type != detail::ShaderBindingType::AccelerationStructure)) {
                    return makeError(Error::InvalidArgument);
                }

                VkDescriptorSetAndBindingMappingEXT mapping{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
                    .descriptorSet = source.descriptorSet,
                    .firstBinding = source.firstBinding,
                    .bindingCount = source.bindingCount,
                    .resourceMask = resourceMask,
                };
                const uint32_t pushOffset = source.pushDataOffset;
                const uint64_t heapOffset =
                    static_cast<uint64_t>(source.heapIndexOffset) * descriptorStride;
                if (heapOffset > UINT32_MAX) {
                    return makeError(Error::InvalidArgument);
                }
                if (source.source == detail::ShaderBindingSource::DeviceAddressFromPushData) {
                    if (source.heapIndexOffset != 0) {
                        return makeError(Error::InvalidArgument);
                    }
                    mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT;
                    mapping.sourceData.pushAddressOffset = pushOffset;
                } else if (source.source == detail::ShaderBindingSource::HeapConstantOffset) {
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

    auto owner = createPipelineOwner<detail::ComputePipelineImpl>(*impl_, desc.usesBindlessHeap);
    if (!owner) { return std::unexpected(owner.error()); }
    auto pipelineImpl = std::move(*owner);
    const VkPipelineLayout layout = pipelineImpl->layout;
    VkResult result = VK_SUCCESS;

    VkPipelineCreateFlags2CreateInfo bindlessPipelineFlags{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .flags = (desc.usesBindlessHeap ? VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT : 0) |
            (desc.indirectBindable ? VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT : 0),
    };
    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = (desc.usesBindlessHeap || desc.indirectBindable) ? static_cast<const void*>(&bindlessPipelineFlags) : nullptr,
        .stage = stage,
        .layout = layout,
    };

#if defined(VK_KHR_pipeline_binary)
    if (impl_->logPipelineKeys) {
        VkPipelineBinaryKeyKHR globalKey{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        VkPipelineBinaryKeyKHR pipelineKey{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        VkPipelineCreateInfoKHR keyInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR,
            .pNext = &pipelineInfo,
        };
        VkResult keyResult = impl_->functions.vkGetPipelineKeyKHR != nullptr
            ? impl_->functions.vkGetPipelineKeyKHR(impl_->device, nullptr, &globalKey)
            : VK_ERROR_EXTENSION_NOT_PRESENT;
        if (keyResult == VK_SUCCESS) {
            keyResult = impl_->functions.vkGetPipelineKeyKHR(impl_->device, &keyInfo, &pipelineKey);
        }
        if (keyResult != VK_SUCCESS || globalKey.keySize == 0 || pipelineKey.keySize == 0 ||
            globalKey.keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR ||
            pipelineKey.keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR) {
            spdlog::error("Pipeline key diagnostic failed: VkResult={}, globalSize={}, pipelineSize={}.",
                static_cast<int>(keyResult), globalKey.keySize, pipelineKey.keySize);
            return makeError(keyResult != VK_SUCCESS ? resultFromVk(keyResult).error() : Error::Failure);
        }
        const auto keyHex = [](const VkPipelineBinaryKeyKHR& key) {
            constexpr char digits[] = "0123456789abcdef";
            std::string text;
            text.reserve(key.keySize * 2);
            for (uint32_t index = 0; index < key.keySize; ++index) {
                text.push_back(digits[key.key[index] >> 4]);
                text.push_back(digits[key.key[index] & 0xf]);
            }
            return text;
        };
        spdlog::info("Vulkan pipeline keys: SPIRV=0x{:016x}, PSO=0x{:016x}, SO={}, global={}, pipeline={}.",
            desc.computeShader.module->impl_->contentHash, psoHash, impl_->capabilities.shaderObject,
            keyHex(globalKey), keyHex(pipelineKey));
    }
#endif

    if (impl_->pipelineExecutableStatistics) {
        if (pipelineInfo.pNext != nullptr) {
            bindlessPipelineFlags.flags |= VK_PIPELINE_CREATE_2_CAPTURE_STATISTICS_BIT_KHR;
        } else {
            pipelineInfo.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        }
    }
    VkPipeline& pipeline = pipelineImpl->pipeline;
    result = impl_->functions.vkCreateComputePipelines(
        impl_->device,
        pipelineCache != nullptr ? pipelineCache->pipelineCache : VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &pipeline);
    if (result != VK_SUCCESS) {
        return std::unexpected(resultFromVk(result).error());
    }

    if (impl_->pipelineExecutableStatistics && impl_->functions.vkGetPipelineExecutablePropertiesKHR && impl_->functions.vkGetPipelineExecutableStatisticsKHR) {
        // The cache key includes metadata and may hash rewritten OMM SPIR-V.
        // Keep both byte fingerprints to join the caller's binding evidence.
        const auto& shader = *desc.computeShader.module->impl_;
        spdlog::info("[PipelineStatisticsBinding] cacheKey={:016x} inputSpirvFnv1a64={} deviceSpirvFnv1a64={} shader={} entry={}",
            shader.contentHash, shader.inputSpirvFnv1a64, shader.deviceSpirvFnv1a64, shader.diagnosticName, stage.pName);
        VkPipelineInfoKHR info{.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipeline};
        uint32_t count = 0;
        VkResult query = impl_->functions.vkGetPipelineExecutablePropertiesKHR(impl_->device, &info, &count, nullptr);
        std::vector<VkPipelineExecutablePropertiesKHR> executables(count, {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
        if (query == VK_SUCCESS && count) {
            query = impl_->functions.vkGetPipelineExecutablePropertiesKHR(impl_->device, &info, &count, executables.data());
        }
        if (query != VK_SUCCESS) { spdlog::warn("[PipelineStatistics] properties unavailable: {}", int(query)); }
        for (uint32_t i = 0; query == VK_SUCCESS && i < count; ++i) {
            VkPipelineExecutableInfoKHR executable{.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pipeline, .executableIndex = i};
            uint32_t statCount = 0;
            VkResult statQuery = impl_->functions.vkGetPipelineExecutableStatisticsKHR(impl_->device, &executable, &statCount, nullptr);
            std::vector<VkPipelineExecutableStatisticKHR> stats(statCount, {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
            if (statQuery == VK_SUCCESS && statCount) {
                statQuery = impl_->functions.vkGetPipelineExecutableStatisticsKHR(impl_->device, &executable, &statCount, stats.data());
            }
            if (statQuery != VK_SUCCESS) { spdlog::warn("[PipelineStatistics] statistics unavailable: {}", int(statQuery)); continue; }
            for (uint32_t j = 0; j < statCount; ++j) {
                const auto& stat = stats[j];
                std::string value;
                switch (stat.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: value = stat.value.b32 ? "true" : "false"; break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: value = std::to_string(stat.value.i64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: value = std::to_string(stat.value.u64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: value = std::to_string(stat.value.f64); break;
                default: value = "unavailable"; break;
                }
                spdlog::info("[PipelineStatistics] entry={} spirv={:016x} executable={} subgroup={} {}={} ({})",
                    stage.pName, desc.computeShader.module->impl_->contentHash, executables[i].name, executables[i].subgroupSize,
                    stat.name, value, stat.description);
            }
        }
    }
    pipelineImpl->replayInputSpirv = desc.computeShader.module->impl_->replayInputSpirv;
    pipelineImpl->replayDeviceSpirv = desc.computeShader.module->impl_->replayDeviceSpirv;
    pipelineImpl->psoHash = psoHash;
    pipelineImpl->pipelineCacheHit = pipelineCache != nullptr &&
        pipelineCache->recordPsoLocked(psoHash);
    return std::unique_ptr<ComputePipeline>(new ComputePipeline(std::move(pipelineImpl)));
}

namespace {

// Both the binary import and export APIs require 16-byte aligned buffers.
struct alignas(16) ShaderBinaryBlock {
    std::array<uint8_t, 16> bytes;
};
using AlignedShaderBinary = std::vector<ShaderBinaryBlock>;
constexpr size_t kMaxShaderBinarySize = 64u * 1024u * 1024u;

uint64_t shaderObjectProgramHash(const GraphicsShaderObjectProgramDesc& desc,
    const std::array<VkShaderCreateInfoEXT, 2>& infos,
    std::span<const VkDescriptorSetAndBindingMappingEXT> mappings)
{
    uint64_t hash = detail::kFnvOffset;
    auto bytes = [&hash](const void* data, size_t size) {
        hash = detail::hashBytes(hash, data, size);
    };
    auto value = [&bytes](uint64_t number) { bytes(&number, sizeof(number)); };
    // Increment when implicit layouts, push ranges or specialization change.
    value(0x534f424a00000001ull);
    value(desc.vertexShader.module->contentHash());
    value(desc.fragmentShader.module->contentHash());
    value(desc.usesBindlessHeap);
    value(desc.bindlessUserPushDataSize);
    for (const auto& info : infos) {
        value(info.flags); value(info.stage); value(info.nextStage);
        const size_t length = std::strlen(info.pName);
        value(length); bytes(info.pName, length);
        // No layouts/push ranges/specialization are currently supplied.
        value(info.setLayoutCount); value(info.pushConstantRangeCount);
        value(info.pNext != nullptr);
    }
    value(mappings.size());
    for (const auto& mapping : mappings) {
        value(mapping.descriptorSet); value(mapping.firstBinding); value(mapping.bindingCount);
        value(mapping.resourceMask); value(mapping.source);
        value(mapping.sourceData.constantOffset.heapOffset);
        value(mapping.sourceData.constantOffset.heapArrayStride);
        value(mapping.sourceData.constantOffset.samplerHeapOffset);
        value(mapping.sourceData.constantOffset.samplerHeapArrayStride);
    }
    return hash;
}

bool exportShaderObjectBinary(detail::DeviceImpl& device, VkShaderEXT shader,
    std::vector<uint8_t>& binary)
{
    if (!device.functions.vkGetShaderBinaryDataEXT) { return false; }
    for (uint32_t attempt = 0; attempt < 3; ++attempt) {
        size_t size = 0;
        VkResult result = device.functions.vkGetShaderBinaryDataEXT(device.device, shader, &size, nullptr);
        if (result != VK_SUCCESS || size == 0 || size > kMaxShaderBinarySize) { return false; }
        AlignedShaderBinary storage((size + 15) / 16);
        result = device.functions.vkGetShaderBinaryDataEXT(device.device, shader, &size, storage.data());
        if (result == VK_INCOMPLETE) { continue; }
        if (result != VK_SUCCESS || size == 0 || size > storage.size() * sizeof(ShaderBinaryBlock)) { return false; }
        const auto* data = reinterpret_cast<const uint8_t*>(storage.data());
        binary.assign(data, data + size);
        return true;
    }
    return false;
}

} // namespace

Result<std::unique_ptr<GraphicsShaderObjectProgram>> Device::createGraphicsShaderObjectProgram(
    const GraphicsShaderObjectProgramDesc& desc)
{
    if (!validShaderStage(desc.vertexShader) || !validShaderStage(desc.fragmentShader)) {
        return makeError(Error::InvalidArgument);
    }

    if (desc.indirectBindable && !supportsIndirectBinding(*impl_,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, IndirectBinding::ShaderObject)) {
        return makeError(Error::Unsupported);
    }
    if (!impl_->capabilities.shaderObject) {
        return makeError(Error::Unsupported);
    }
    if (desc.usesBindlessHeap) {
        if (!impl_->capabilities.bindlessDescriptorHeap) {
            return makeError(Error::Unsupported);
        }
        const VkDeviceSize requiredPushDataSize =
            desc.bindlessUserPushDataSize;
        if (impl_->descriptorHeapWriter.maxPushDataSize() < requiredPushDataSize) {
            return makeError(Error::Unsupported);
        }
    }

    std::array<VkDescriptorSetAndBindingMappingEXT, 3> bindlessMappings{};
    VkShaderDescriptorSetAndBindingMappingInfoEXT bindlessMappingInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
    };
    if (desc.usesBindlessHeap) {
        bindlessMappings = defaultHeapMappings(impl_->descriptorHeapWriter);
        bindlessMappingInfo.mappingCount = static_cast<uint32_t>(bindlessMappings.size());
        bindlessMappingInfo.pMappings = bindlessMappings.data();
    }

    const auto& vertex = *desc.vertexShader.module->impl_;
    const auto& fragment = *desc.fragmentShader.module->impl_;
    const VkShaderCreateFlagsEXT shaderFlags =
        VK_SHADER_CREATE_LINK_STAGE_BIT_EXT |
        (desc.usesBindlessHeap ? VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT : 0) |
        (desc.indirectBindable ? VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT : 0);
    std::array<VkShaderCreateInfoEXT, 2> shaderInfos{
        VkShaderCreateInfoEXT{
            .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
            .pNext = desc.usesBindlessHeap && vertex.hasDescriptorBindings
                ? static_cast<const void*>(&bindlessMappingInfo) : nullptr,
            .flags = shaderFlags,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
            .codeSize = static_cast<size_t>(vertex.deviceSpirv.size() * sizeof(uint32_t)),
            .pCode = vertex.deviceSpirv.data(),
            .pName = desc.vertexShader.entryPoint,
        },
        VkShaderCreateInfoEXT{
            .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
            .pNext = desc.usesBindlessHeap && fragment.hasDescriptorBindings
                ? static_cast<const void*>(&bindlessMappingInfo) : nullptr,
            .flags = shaderFlags,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .nextStage = 0,
            .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
            .codeSize = static_cast<size_t>(fragment.deviceSpirv.size() * sizeof(uint32_t)),
            .pCode = fragment.deviceSpirv.data(),
            .pName = desc.fragmentShader.entryPoint,
        },
    };

    std::array<VkShaderEXT, 2> shaders{
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
    };
    ShaderObjectCacheStats stats;
    stats.programHash = shaderObjectProgramHash(desc, shaderInfos,
        desc.usesBindlessHeap ? std::span<const VkDescriptorSetAndBindingMappingEXT>(bindlessMappings) :
            std::span<const VkDescriptorSetAndBindingMappingEXT>{});
    detail::ShaderObjectCacheFileIdentity identity{.binaryVersion = impl_->physicalProperties.shaderObject.shaderBinaryVersion,
        .programHash = stats.programHash};
    std::copy_n(impl_->physicalProperties.shaderObject.shaderBinaryUUID, identity.binaryUUID.size(), identity.binaryUUID.begin());
    std::filesystem::path cachePath;
    std::unique_lock<std::mutex> cacheLock;
    if (desc.binaryCacheDirectory && desc.binaryCacheDirectory[0]) {
        std::string uuid;
        for (const uint8_t byte : identity.binaryUUID) { uuid += fmt::format("{:02x}", byte); }
        cachePath = std::filesystem::path(desc.binaryCacheDirectory) / uuid /
            fmt::format("{:016x}.shaderbin", stats.programHash);
        cacheLock = std::unique_lock(impl_->shaderObjectCacheMutexes[stats.programHash % impl_->shaderObjectCacheMutexes.size()]);
    }
    auto destroyShaders = [&] {
        for (VkShaderEXT shader : shaders) {
            if (shader != VK_NULL_HANDLE) {
                impl_->functions.vkDestroyShaderEXT(impl_->device, shader, nullptr);
            }
        }
        shaders.fill(VK_NULL_HANDLE);
    };
    auto createShaders = [&](const auto& infos) {
        const auto begin = std::chrono::steady_clock::now();
        const VkResult result = impl_->functions.vkCreateShadersEXT(impl_->device,
            static_cast<uint32_t>(infos.size()), infos.data(), nullptr, shaders.data());
        stats.creationTimeNanoseconds += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
        return result;
    };
    if (!cachePath.empty()) {
        detail::ShaderObjectCacheFileData cached;
        std::string reason;
        const auto status = detail::loadShaderObjectCacheFile(cachePath, identity, cached, reason);
        stats.loadStatus = static_cast<PipelineCacheLoadStatus>(status);
        if (status == detail::ShaderObjectCacheFileLoadStatus::Loaded) {
            auto binaryInfos = shaderInfos;
            std::array<AlignedShaderBinary, 2> storage;
            for (size_t i = 0; i < storage.size(); ++i) {
                storage[i].resize((cached.binaries[i].size() + 15) / 16);
                std::memcpy(storage[i].data(), cached.binaries[i].data(), cached.binaries[i].size());
                binaryInfos[i].codeType = VK_SHADER_CODE_TYPE_BINARY_EXT;
                binaryInfos[i].codeSize = cached.binaries[i].size();
                binaryInfos[i].pCode = storage[i].data();
                // Descriptor mappings are baked into the exported binary.
                binaryInfos[i].pNext = nullptr;
            }
            const VkResult result = createShaders(binaryInfos);
            if (result == VK_SUCCESS) {
                stats.binaryCacheHit = true;
                stats.persisted = true;
                stats.binaryDataSize = cached.binaries[0].size() + cached.binaries[1].size();
            } else {
                destroyShaders();
                stats.driverRejected = true;
                spdlog::warn("[ShaderObjectCache] Driver rejected {} (VkResult {}); rebuilding from SPIR-V", cachePath.string(), static_cast<int>(result));
            }
        } else if (status != detail::ShaderObjectCacheFileLoadStatus::NotFound) {
            spdlog::warn("[ShaderObjectCache] Ignoring {}: {}", cachePath.string(), reason);
        }
    }
    if (!stats.binaryCacheHit) {
        const VkResult result = createShaders(shaderInfos);
        if (result != VK_SUCCESS) {
            destroyShaders();
            return std::unexpected(resultFromVk(result).error());
        }
        if (!cachePath.empty()) {
            detail::ShaderObjectCacheFileData data;
            if (exportShaderObjectBinary(*impl_, shaders[0], data.binaries[0]) &&
                exportShaderObjectBinary(*impl_, shaders[1], data.binaries[1])) {
                stats.binaryDataSize = data.binaries[0].size() + data.binaries[1].size();
                std::string reason;
                stats.persisted = detail::saveShaderObjectCacheFile(cachePath, identity, data, reason);
                if (!stats.persisted) { spdlog::warn("[ShaderObjectCache] Could not save {}: {}", cachePath.string(), reason); }
            } else {
                spdlog::warn("[ShaderObjectCache] Could not export both stages for {}; current shader remains usable", cachePath.string());
            }
        }
    }
    vulkan::toolingHooks().shaderBinary(vertex.deviceSpirv.data(), vertex.deviceSpirv.size() * sizeof(uint32_t));
    vulkan::toolingHooks().shaderBinary(fragment.deviceSpirv.data(), fragment.deviceSpirv.size() * sizeof(uint32_t));

    auto programImpl = std::make_unique<detail::GraphicsShaderObjectProgramImpl>();
    programImpl->device = impl_.get();
    programImpl->vertexShader = shaders[0];
    programImpl->fragmentShader = shaders[1];
    programImpl->usesBindlessHeap = desc.usesBindlessHeap;
    programImpl->cacheStats = stats;
    programImpl->binaryCacheFilePath = cachePath.string();
    return std::unique_ptr<GraphicsShaderObjectProgram>(new GraphicsShaderObjectProgram(std::move(programImpl)));
}

Result<std::unique_ptr<BindlessHeap>> Device::createBindlessHeap(const BindlessHeapDesc& desc)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    if (!impl_->capabilities.bindlessDescriptorHeap) {
        return makeError(Error::Unsupported);
    }

    auto bindlessImpl = std::make_unique<detail::BindlessHeapImpl>();
    Result<> result = bindlessImpl->initialize(*impl_, desc);
    if (!result) {
        return std::unexpected(result.error());
    }

    return std::unique_ptr<BindlessHeap>(new BindlessHeap(std::move(bindlessImpl)));
}

Result<std::unique_ptr<Device>> createDevice(const DeviceDesc& desc)
{
    const auto* suppliedExtensions = std::any_cast<vulkan::VulkanDeviceExtensions>(&desc.backendExtensions);
    if (desc.backendExtensions.has_value() && !suppliedExtensions) {
        spdlog::error("Vulkan createDevice requires VulkanDeviceExtensions or an empty backendExtensions value.");
        return makeError(Error::InvalidArgument);
    }
    // Snapshot options before invoking loaders, callbacks or optional integrations.
    const auto vulkanOptions = suppliedExtensions ? *suppliedExtensions : vulkan::VulkanDeviceExtensions{};

    if (!desc.enableShaderObject) {
        spdlog::error("Shader Object is required: DeviceDesc::enableShaderObject must be true.");
        return makeError(Error::InvalidArgument);
    }

    const char* internalPipelineCacheMode = std::getenv("METALLIC_VK_INTERNAL_PIPELINE_CACHE");
    const bool diagnoseInternalPipelineCache = internalPipelineCacheMode != nullptr;
    const bool disableInternalPipelineCache = diagnoseInternalPipelineCache &&
        std::strcmp(internalPipelineCacheMode, "disabled") == 0;
    if (diagnoseInternalPipelineCache && !disableInternalPipelineCache &&
        std::strcmp(internalPipelineCacheMode, "enabled") != 0) {
        spdlog::error("METALLIC_VK_INTERNAL_PIPELINE_CACHE must be enabled or disabled.");
        return makeError(Error::InvalidArgument);
    }
    const char* logPipelineKeysValue = std::getenv("METALLIC_VK_LOG_PIPELINE_KEYS");
    const bool logPipelineKeys = logPipelineKeysValue != nullptr && std::strcmp(logPipelineKeysValue, "1") == 0;
    if (logPipelineKeys && !diagnoseInternalPipelineCache) {
        spdlog::error("METALLIC_VK_LOG_PIPELINE_KEYS requires METALLIC_VK_INTERNAL_PIPELINE_CACHE=enabled or disabled.");
        return makeError(Error::InvalidArgument);
    }

    std::lock_guard initializationLock(volkInitializationMutex());
    auto deviceImpl = std::make_unique<detail::DeviceImpl>();
    deviceImpl->logPipelineKeys = logPipelineKeys;
    if (desc.enableAftermath) {
        vulkan::toolingHooks().initializeDiagnostics(desc.applicationName);
    }

    PFN_vkGetInstanceProcAddr streamlineVkGetInstanceProcAddr = nullptr;
    if (desc.enableStreamline && vulkan::streamlineSdkAvailable()) {
        const char* const vulkanLibraryName = vulkan::streamlineVulkanLibraryName();
        streamlineVkGetInstanceProcAddr =
            loadVulkanLoaderProcAddr(vulkanLibraryName, deviceImpl->vulkanLoaderHandle);
        if (streamlineVkGetInstanceProcAddr != nullptr) {
            std::string streamlineLog;
            Result<> streamlineResult = vulkan::initializeStreamlinePreDevice(streamlineLog);
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
            return std::unexpected(resultFromVk(vkResult).error());
        }
    }

    deviceImpl->getInstanceProcAddr = vkGetInstanceProcAddr;
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
    if (hasName(availableExtensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) &&
        std::none_of(instanceExtensions.begin(), instanceExtensions.end(), [](const char* name) {
            return std::strcmp(name, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) == 0;
        })) {
        instanceExtensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    }
    const bool debugUtilsAvailable = hasName(availableExtensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugUtilsAvailable) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        deviceImpl->debugUtilsEnabled = true;
    }

    std::string toolingError;
    if (!vulkan::toolingHooks().instanceExtensions(instanceExtensions, kVulkanAPIVersion,
            desc.enableValidation || desc.enableSynchronizationValidation || vulkanOptions.shaderPrintf != nullptr, toolingError)) {
        spdlog::error("[Vulkan tooling] {}", toolingError);
        return makeError(Error::Unsupported);
    }
    std::vector<const char*> instanceLayers;
    const std::vector<VkLayerProperties> availableLayers = enumerateInstanceLayers();
    const bool validationRequested = desc.enableValidation || desc.enableSynchronizationValidation || vulkanOptions.shaderPrintf != nullptr;
    if (vulkanOptions.shaderPrintf) {
        auto& printf = *vulkanOptions.shaderPrintf;
        for (const auto& layer : availableLayers) {
            if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                printf.layerDiscovered = true;
                printf.layerSpecVersion = layer.specVersion;
                printf.layerImplementationVersion = layer.implementationVersion;
            }
        }
        if (!printf.valid() || !printf.layerDiscovered || !debugUtilsAvailable) {
            spdlog::error("Shader Printf requires a valid buffer budget, Khronos validation and debug utils.");
            return makeError(Error::Unsupported);
        }
        if (!printf.options().settingsViaFile) {
            uint32_t count = 0;
            vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &count, nullptr);
            std::vector<VkExtensionProperties> layerExtensions(count);
            if (vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &count, layerExtensions.data()) != VK_SUCCESS ||
                (!hasName(layerExtensions, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME) &&
                 !hasName(availableExtensions, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME))) {
                spdlog::error("Shader Printf requires VK_EXT_layer_settings.");
                return makeError(Error::Unsupported);
            }
            instanceExtensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
        }
    }
    if (validationRequested && hasName(availableLayers, "VK_LAYER_KHRONOS_validation")) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
        deviceImpl->validationEnabled = true;
    } else if (validationRequested) {
        spdlog::warn("Vulkan validation requested but VK_LAYER_KHRONOS_validation is not available.");
    }

    // Keep this mode explicit and fail closed; ShaderPrintf owns a different
    // validation configuration and cannot be combined with conformance checks.
    if (desc.enableSynchronizationValidation) {
        if (!deviceImpl->validationEnabled || !debugUtilsAvailable || vulkanOptions.shaderPrintf) {
            return makeError(Error::Unsupported);
        }
        uint32_t count = 0;
        if (vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &count, nullptr) != VK_SUCCESS) {
            return makeError(Error::Unsupported);
        }
        std::vector<VkExtensionProperties> layerExtensions(count);
        if (vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &count, layerExtensions.data()) != VK_SUCCESS ||
            (!hasName(layerExtensions, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME) &&
             !hasName(availableExtensions, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME))) {
            return makeError(Error::Unsupported);
        }
        instanceExtensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
    }
    const VkBool32 syncValidation = VK_TRUE;
    const VkLayerSettingEXT syncSetting{"VK_LAYER_KHRONOS_validation", "validate_sync",
        VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &syncValidation};
    const VkLayerSettingsCreateInfoEXT syncSettings{VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        nullptr, 1, &syncSetting};

    VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = desc.applicationName,
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Metallic",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = kVulkanAPIVersion,
    };

    deviceImpl->debugContext = {desc.validationSink, vulkanOptions.shaderPrintf};
    VkDebugUtilsMessengerCreateInfoEXT earlyMessages{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = desc.enableSynchronizationValidation ? &syncSettings : vulkanOptions.shaderPrintf ? vulkanOptions.shaderPrintf->settings() : nullptr,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData = &deviceImpl->debugContext,
    };
    if (vulkanOptions.shaderPrintf && vulkanOptions.shaderPrintf->options().subscribeInfo) {
        earlyMessages.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
    }
    VkInstanceCreateInfo instanceInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = desc.enableSynchronizationValidation || vulkanOptions.shaderPrintf || (validationRequested && debugUtilsAvailable && desc.validationSink.callback)
            ? &earlyMessages : nullptr,
        .pApplicationInfo = &applicationInfo,
        .enabledLayerCount = static_cast<uint32_t>(instanceLayers.size()),
        .ppEnabledLayerNames = instanceLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
        .ppEnabledExtensionNames = instanceExtensions.data(),
    };

    vkResult = vkCreateInstance(&instanceInfo, nullptr, &deviceImpl->instance);
    if (vkResult != VK_SUCCESS) {
        return std::unexpected(resultFromVk(vkResult).error());
    }
    volkLoadInstanceOnly(deviceImpl->instance);
    volkLoadInstanceTable(&deviceImpl->instanceFunctions, deviceImpl->instance);

    if (deviceImpl->validationEnabled && debugUtilsAvailable) {
        deviceImpl->debugMessenger = createDebugMessenger(deviceImpl->instance, &deviceImpl->debugContext);
    }
    if (desc.enableSynchronizationValidation) {
        if (!deviceImpl->debugMessenger) { return makeError(Error::Unsupported); }
        deviceImpl->synchronizationValidationEnabled = true;
    }
    if (vulkanOptions.shaderPrintf) {
        vulkanOptions.shaderPrintf->instanceConfigured = true;
        vulkanOptions.shaderPrintf->messengerConfigured = deviceImpl->debugMessenger != VK_NULL_HANDLE;
        if (!vulkanOptions.shaderPrintf->messengerConfigured) { return makeError(Error::Unsupported); }
    }

    uint32_t physicalDeviceCount = 0;
    vkResult = vkEnumeratePhysicalDevices(deviceImpl->instance, &physicalDeviceCount, nullptr);
    if (vkResult != VK_SUCCESS || physicalDeviceCount == 0) {
        if (vkResult == VK_SUCCESS) {
            return makeError(Error::Unsupported);
        }
        return std::unexpected(resultFromVk(vkResult).error());
    }

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(deviceImpl->instance, &physicalDeviceCount, physicalDevices.data());

    const VulkanDeviceFeatureRequest requestedFeatures = VulkanDeviceFeatureRequest::from(desc, vulkanOptions, vulkan::toolingHooks().diagnosticsInitialized());
    bool validationSupportsOpacityMicromap = true;
    if (deviceImpl->validationEnabled && !vulkan::toolingHooks().captureInjected()) {
        for (const auto& layer : availableLayers) {
            if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0 &&
                layer.specVersion < VK_MAKE_API_VERSION(0, 1, 4, 357)) {
                validationSupportsOpacityMicromap = false;
            }
        }
        if (!validationSupportsOpacityMicromap && requestedFeatures.opacityMicromap &&
            (requestedFeatures.rayTracingAccelerationStructure || requestedFeatures.rayQuery || requestedFeatures.streamline)) {
            spdlog::warn("[Vulkan] KHR OMM needs validation layers 1.4.357 or newer. "
                "Keeping shader alpha traversal while this older validation layer is enabled.");
        }
    }
    VulkanPhysicalDeviceCandidate bestCandidate;
    VulkanDeviceFeatureSelection selectedFeatures;

    for (VkPhysicalDevice physicalDevice : physicalDevices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        if (properties.apiVersion < kVulkanAPIVersion) {
            continue;
        }

        VulkanExtensionSet extensions = queryDeviceExtensions(physicalDevice);
        if (!extensions.opacityMicromapExt) {
            extensions.opacityMicromap &= validationSupportsOpacityMicromap;
        }
        if (!extensions.deviceAddressCommands) {
            spdlog::warn("Skipping Vulkan device '{}': VK_KHR_device_address_commands is unavailable.",
                properties.deviceName);
            continue;
        }
        if (!extensions.swapchain || !extensions.shaderObject) {
            continue;
        }

        VulkanDeviceFeatureProbe probe;
        probe.buildChain(extensions);
        vkGetPhysicalDeviceFeatures2(physicalDevice, &probe.features);
        vkGetPhysicalDeviceProperties2(physicalDevice, &probe.properties);
        if (vulkanOptions.shaderPrintf && (!probe.features.features.fragmentStoresAndAtomics ||
            !probe.features.features.vertexPipelineStoresAndAtomics ||
            !probe.vulkan12Features.vulkanMemoryModel || !probe.vulkan12Features.vulkanMemoryModelDeviceScope ||
            !probe.vulkan12Features.storageBuffer8BitAccess || !probe.vulkan12Features.shaderInt8 ||
            !probe.vulkan11Features.storageBuffer16BitAccess)) { continue; }
        if (probe.vulkan12Features.bufferDeviceAddress != VK_TRUE ||
            probe.deviceAddressCommandsFeatures.deviceAddressCommands != VK_TRUE) {
            spdlog::warn("Skipping Vulkan device '{}': deviceAddressCommands and bufferDeviceAddress must be supported.",
                properties.deviceName);
            continue;
        }
        if (!probe.supportsRequiredCoreFeatures() ||
            probe.shaderObjectFeatures.shaderObject != VK_TRUE) {
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

        if (desc.enableAsyncCompute) {
            for (uint32_t queueIndex = 0; queueIndex < queueFamilyCount; ++queueIndex) {
                const VkQueueFlags flags = queueFamilies[queueIndex].queueFlags;
                if (queueFamilies[queueIndex].queueCount != 0 &&
                    (flags & VK_QUEUE_COMPUTE_BIT) != 0 && (flags & VK_QUEUE_GRAPHICS_BIT) == 0) {
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
            requestedFeatures, extensions, probe,
            requestedFeatures.bindlessDescriptorHeap && extensions.descriptorHeap &&
                probe.descriptorHeapFeatures.descriptorHeap == VK_TRUE &&
                DescriptorHeapWriter::hasUsableProperties(physicalDevice));
        const int32_t featureScore =
            featureSelection.score(requestedFeatures);
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
        spdlog::error(
            "No suitable Vulkan device: required core features, graphics/compute queues, "
            "VK_KHR_swapchain, VK_EXT_shader_object with shaderObject=true, and "
            "VK_KHR_device_address_commands with deviceAddressCommands=true and bufferDeviceAddress=true are required.");
        return makeError(Error::Unsupported);
    }

    deviceImpl->physicalProperties = vulkan::queryDeviceProperties(
        deviceImpl->physicalDevice, selectedFeatures, deviceImpl->instanceFunctions.vkGetPhysicalDeviceProperties2);

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

    deviceImpl->computeQueueIndex = desc.enableAsyncCompute &&
        deviceImpl->computeFamily == deviceImpl->graphicsFamily &&
        selectedQueueFamilies[deviceImpl->computeFamily].queueCount > 1 ? 1u : 0u;
    deviceImpl->copyQueueIndex = UINT32_MAX;
    if (deviceImpl->copyFamily < selectedQueueFamilies.size()) {
        const bool copyUsesExistingQueue =
            deviceImpl->copyFamily == deviceImpl->graphicsFamily ||
            deviceImpl->copyFamily == deviceImpl->computeFamily;
        if (!copyUsesExistingQueue) {
            deviceImpl->copyQueueIndex = 0;
        } else {
            const uint32_t used = deviceImpl->copyFamily == deviceImpl->computeFamily
                ? deviceImpl->computeQueueIndex + 1u : 1u;
            if (selectedQueueFamilies[deviceImpl->copyFamily].queueCount > used) { deviceImpl->copyQueueIndex = used; }
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
    requestQueues(deviceImpl->computeFamily, deviceImpl->computeQueueIndex + 1u);
    if (deviceImpl->copyQueueIndex != UINT32_MAX) {
        requestQueues(deviceImpl->copyFamily, deviceImpl->copyQueueIndex + 1);
    }

    const std::array<float, 3> queuePriorities{1.0f, 1.0f, 1.0f};
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
    configureAftermathDiagnostics(enabledFeatureChain, selectedFeatures);
    if (vulkanOptions.shaderPrintf) {
        enabledFeatureChain.features.features.fragmentStoresAndAtomics = VK_TRUE;
        enabledFeatureChain.features.features.vertexPipelineStoresAndAtomics = VK_TRUE;
        enabledFeatureChain.vulkan12Features.vulkanMemoryModel = VK_TRUE;
        enabledFeatureChain.vulkan12Features.vulkanMemoryModelDeviceScope = VK_TRUE;
        enabledFeatureChain.vulkan12Features.storageBuffer8BitAccess = VK_TRUE;
        enabledFeatureChain.vulkan12Features.shaderInt8 = VK_TRUE;
        enabledFeatureChain.vulkan11Features.storageBuffer16BitAccess = VK_TRUE;
    }
    std::vector<const char*> deviceExtensions = enabledDeviceExtensions(selectedFeatures);
    const VulkanExtensionSet selectedDeviceExtensions = queryDeviceExtensions(deviceImpl->physicalDevice);
    deviceImpl->hdrMetadataExtension = selectedDeviceExtensions.has(VK_EXT_HDR_METADATA_EXTENSION_NAME);
    if (deviceImpl->hdrMetadataExtension) { deviceExtensions.push_back(VK_EXT_HDR_METADATA_EXTENSION_NAME); }
    deviceImpl->memoryBudgetExtension = selectedDeviceExtensions.has(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (deviceImpl->memoryBudgetExtension && std::none_of(deviceExtensions.begin(), deviceExtensions.end(), [](const char* name) {
            return std::strcmp(name, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
        })) {
        deviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
    vkGetPhysicalDeviceMemoryProperties(deviceImpl->physicalDevice, &deviceImpl->memoryProperties);
    deviceImpl->memoryBudgetState->policy = desc.memoryBudget;
#if defined(VK_NV_low_latency2) && defined(VK_KHR_present_id)
    // Streamline's Reflex plugin can inject low_latency2 into vkCreateDevice.
    // Supply its present-id dependency when the adapter exposes that extension.
    if (deviceImpl->streamlineInitialized && selectedDeviceExtensions.has(VK_NV_LOW_LATENCY_2_EXTENSION_NAME)) {
        if (selectedDeviceExtensions.has(VK_KHR_PRESENT_ID_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME);
        }
#ifdef VK_KHR_present_id2
        else if (selectedDeviceExtensions.has(VK_KHR_PRESENT_ID_2_EXTENSION_NAME)) {
            deviceExtensions.push_back(VK_KHR_PRESENT_ID_2_EXTENSION_NAME);
        }
#endif
    }
#endif
    // Calibration is optional; never reject a GPU just because it lacks it.
    const bool calibratedTimestamps = selectedDeviceExtensions.has(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    if (calibratedTimestamps) {
        deviceExtensions.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    }
    const void* deviceCreateNext = &enabledFeatureChain.features;
#if defined(VK_KHR_pipeline_binary)
    VkPhysicalDevicePipelineBinaryFeaturesKHR pipelineBinaryFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR,
    };
    VkDevicePipelineBinaryInternalCacheControlKHR internalCacheControl{
        .sType = VK_STRUCTURE_TYPE_DEVICE_PIPELINE_BINARY_INTERNAL_CACHE_CONTROL_KHR,
    };
    if (diagnoseInternalPipelineCache) {
        const VulkanExtensionSet extensions = queryDeviceExtensions(deviceImpl->physicalDevice);
        if (!extensions.has(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME)) {
            spdlog::error("Internal pipeline cache diagnostic requires VK_KHR_pipeline_binary.");
            return makeError(Error::Unsupported);
        }

        VkPhysicalDeviceFeatures2 binaryFeatureProbe{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &pipelineBinaryFeatures,
        };
        vkGetPhysicalDeviceFeatures2(deviceImpl->physicalDevice, &binaryFeatureProbe);
        VkPhysicalDevicePipelineBinaryPropertiesKHR pipelineBinaryProperties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_PROPERTIES_KHR,
        };
        VkPhysicalDeviceProperties2 binaryPropertyProbe{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &pipelineBinaryProperties,
        };
        vkGetPhysicalDeviceProperties2(deviceImpl->physicalDevice, &binaryPropertyProbe);
        if (pipelineBinaryFeatures.pipelineBinaries != VK_TRUE ||
            pipelineBinaryProperties.pipelineBinaryInternalCacheControl != VK_TRUE) {
            spdlog::error(
                "Internal pipeline cache diagnostic unsupported: pipelineBinaries={}, cacheControl={}.",
                pipelineBinaryFeatures.pipelineBinaries,
                pipelineBinaryProperties.pipelineBinaryInternalCacheControl);
            return makeError(Error::Unsupported);
        }

        // Keep the extension and feature chain identical in both diagnostic modes.
        // Only the cache-control value changes; an unset environment keeps the default path.
        deviceExtensions.push_back(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME);
        pipelineBinaryFeatures.pNext = &enabledFeatureChain.features;
        internalCacheControl.pNext = &pipelineBinaryFeatures;
        internalCacheControl.disableInternalCache = disableInternalPipelineCache ? VK_TRUE : VK_FALSE;
        deviceCreateNext = &internalCacheControl;
        spdlog::info(
            "Internal pipeline cache diagnostic: mode={}, internalCache={}, cacheControl={}.",
            internalPipelineCacheMode,
            pipelineBinaryProperties.pipelineBinaryInternalCache,
            pipelineBinaryProperties.pipelineBinaryInternalCacheControl);
    }
#else
    if (diagnoseInternalPipelineCache) {
        spdlog::error("Internal pipeline cache diagnostic requires headers with VK_KHR_pipeline_binary.");
        return makeError(Error::Unsupported);
    }
#endif
    // Opt-in compiler resource diagnostics; normal devices/pipelines are unchanged.
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executableFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
    };
    const char* executableStats = std::getenv("METALLIC_VK_PIPELINE_STATISTICS");
    if (executableStats != nullptr && std::strcmp(executableStats, "1") == 0) {
        if (selectedDeviceExtensions.has(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
            VkPhysicalDeviceFeatures2 probe{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &executableFeatures};
            vkGetPhysicalDeviceFeatures2(deviceImpl->physicalDevice, &probe);
            if (executableFeatures.pipelineExecutableInfo) {
                deviceExtensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
                executableFeatures.pNext = const_cast<void*>(deviceCreateNext);
                deviceCreateNext = &executableFeatures;
                deviceImpl->pipelineExecutableStatistics = true;
            }
        }
        spdlog::info("[PipelineStatistics] enabled={}", deviceImpl->pipelineExecutableStatistics);
    }
    if (!vulkan::toolingHooks().deviceExtensions(deviceImpl->instance, deviceImpl->physicalDevice, deviceExtensions, toolingError)) {
        spdlog::error("[Vulkan tooling] {}", toolingError);
        return makeError(Error::Unsupported);
    }
    VkDeviceCreateInfo deviceInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = deviceCreateNext,
        .queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size()),
        .pQueueCreateInfos = queueInfos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };

    vkResult = vkCreateDevice(deviceImpl->physicalDevice, &deviceInfo, nullptr, &deviceImpl->device);
    if (vkResult != VK_SUCCESS) {
        return std::unexpected(resultFromVk(vkResult).error());
    }

    volkLoadDeviceTable(&deviceImpl->functions, deviceImpl->device);
    if (vulkanOptions.shaderPrintf) { vulkanOptions.shaderPrintf->deviceConfigured = true; }

    const auto& selectedProperties = deviceImpl->physicalProperties.core;
    if (calibratedTimestamps) {
        const auto getDomains = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(
            vkGetInstanceProcAddr(deviceImpl->instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
        deviceImpl->getCalibratedTimestamps = reinterpret_cast<PFN_vkGetCalibratedTimestampsEXT>(
            vkGetDeviceProcAddr(deviceImpl->device, "vkGetCalibratedTimestampsEXT"));
        uint32_t count = 0;
        if (getDomains != nullptr && getDomains(deviceImpl->physicalDevice, &count, nullptr) == VK_SUCCESS) {
            std::vector<VkTimeDomainEXT> domains(count);
            if (getDomains(deviceImpl->physicalDevice, &count, domains.data()) == VK_SUCCESS) {
#if defined(_WIN32)
                constexpr auto hostDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#elif defined(__linux__)
                constexpr auto hostDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
#else
                constexpr auto hostDomain = VK_TIME_DOMAIN_DEVICE_EXT;
#endif
                if (std::find(domains.begin(), domains.end(), hostDomain) != domains.end()) {
                    deviceImpl->calibrationHostDomain = hostDomain;
                }
            }
        }
    }
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
        vkResult = deviceImpl->descriptorHeapWriter.initialize(deviceImpl->physicalProperties.descriptorHeap, deviceImpl->device, deviceImpl->functions);
        if (vkResult != VK_SUCCESS) {
            return std::unexpected(resultFromVk(vkResult).error());
        }

        const VkDeviceSize resourceShaderStride = deviceImpl->descriptorHeapWriter.resourceDescriptorStride();
        const VkDeviceSize samplerShaderStride = deviceImpl->descriptorHeapWriter.samplerDescriptorSize();
        if (resourceShaderStride > std::numeric_limits<int32_t>::max() ||
            samplerShaderStride > std::numeric_limits<int32_t>::max()) {
            spdlog::error("Vulkan descriptor heap strides exceed Slang's compiler option range.");
            return makeError(Error::Unsupported);
        }
        // TO-REMOVE(VVL payload-size): emit literal strides in Slang while VVL
        // miscalculates task/mesh payloads with unresolved opaque-size queries.
        setSlangDescriptorHeapShaderStrides({static_cast<uint32_t>(resourceShaderStride),
            static_cast<uint32_t>(samplerShaderStride)});

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
            deviceImpl->descriptorHeapWriter.resourceDescriptorStride());
        deviceImpl->capabilities.maxBindlessBuffers = capacityFromBytes(
            resourceCapacityBytes,
            deviceImpl->descriptorHeapWriter.resourceDescriptorStride());
        deviceImpl->bindlessDescriptorHeapEnabled = true;
    }
    selectedFeatures.publish(deviceImpl->capabilities, deviceImpl->vulkanCapabilities);
    deviceImpl->capabilities.memoryDecompression = selectedFeatures.memoryDecompression && deviceImpl->functions.vkCmdDecompressMemoryEXT != nullptr;
    deviceImpl->shaderUntypedPointersEnabled = selectedFeatures.shaderUntypedPointers;
    deviceImpl->rayTracingPipelineEnabled = selectedFeatures.streamline;
    deviceImpl->opacityMicromapExt = selectedFeatures.opacityMicromapExt;
    if (selectedFeatures.opacityMicromap) {
        spdlog::info("[Vulkan] {} enabled{}",
            selectedFeatures.opacityMicromapExt ? VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME : VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME,
            selectedFeatures.opacityMicromapExt ? " (Nsight Graphics workaround)" : "");
    }
    if (selectedFeatures.rayTracingPositionFetch) {
        spdlog::info("[Vulkan] VK_KHR_ray_tracing_position_fetch enabled");
    }
    deviceImpl->capabilities.subPixelPrecisionBits = selectedProperties.limits.subPixelPrecisionBits;
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
    allocatorInfo.vulkanApiVersion = kVulkanAPIVersion;
    // Vulkan 1.4 exposes usage2; tell VMA to inspect it instead of legacy usage.
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
    if (deviceImpl->memoryBudgetExtension) { allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT; }
    if (deviceImpl->bufferDeviceAddressEnabled) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    VmaVulkanFunctions vulkanFunctions{};
    vkResult = vmaImportVulkanFunctionsFromVolk(&allocatorInfo, &vulkanFunctions);
    if (vkResult != VK_SUCCESS) {
        return std::unexpected(resultFromVk(vkResult).error());
    }
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;
    vkResult = vmaCreateAllocator(&allocatorInfo, &deviceImpl->allocator);
    if (vkResult != VK_SUCCESS) {
        return std::unexpected(resultFromVk(vkResult).error());
    }

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    deviceImpl->functions.vkGetDeviceQueue(deviceImpl->device, deviceImpl->graphicsFamily, 0, &graphicsQueue);
    deviceImpl->addQueue(
        graphicsQueue,
        deviceImpl->graphicsFamily,
        selectedQueueFamilies[deviceImpl->graphicsFamily].queueFlags,
        selectedQueueFamilies[deviceImpl->graphicsFamily].timestampValidBits,
        QueueType::Graphics);

    VkQueue computeQueue = VK_NULL_HANDLE;
    deviceImpl->functions.vkGetDeviceQueue(deviceImpl->device, deviceImpl->computeFamily, deviceImpl->computeQueueIndex, &computeQueue);
    deviceImpl->capabilities.independentComputeQueue = computeQueue != graphicsQueue;
    if (deviceImpl->capabilities.independentComputeQueue) {
        spdlog::info("[Vulkan] Independent compute queue enabled: graphics family {}, compute family {} index {}",
            deviceImpl->graphicsFamily, deviceImpl->computeFamily, deviceImpl->computeQueueIndex);
    }
    deviceImpl->addQueue(
        computeQueue,
        deviceImpl->computeFamily,
        selectedQueueFamilies[deviceImpl->computeFamily].queueFlags,
        selectedQueueFamilies[deviceImpl->computeFamily].timestampValidBits,
        QueueType::Compute);

    if (deviceImpl->copyQueueIndex != UINT32_MAX) {
        VkQueue copyQueue = VK_NULL_HANDLE;
        deviceImpl->functions.vkGetDeviceQueue(
            deviceImpl->device,
            deviceImpl->copyFamily,
            deviceImpl->copyQueueIndex,
            &copyQueue);
        if (copyQueue != VK_NULL_HANDLE && copyQueue != graphicsQueue && copyQueue != computeQueue) {
            deviceImpl->addQueue(
                copyQueue,
                deviceImpl->copyFamily,
                selectedQueueFamilies[deviceImpl->copyFamily].queueFlags,
                selectedQueueFamilies[deviceImpl->copyFamily].timestampValidBits,
                QueueType::Copy);
            deviceImpl->capabilities.independentCopyQueue = true;
        } else {
            deviceImpl->copyQueueIndex = UINT32_MAX;
        }
    }

    if (deviceImpl->streamlineInitialized && selectedFeatures.streamline) {
        std::string streamlineLog;
        Result<> streamlineResult = setStreamlineVulkanDevice(
            vulkan::NativeDevice{
                .functions = &deviceImpl->functions,
                .instanceFunctions = &deviceImpl->instanceFunctions,
                .properties = &deviceImpl->physicalProperties,
                .getInstanceProcAddr = deviceImpl->getInstanceProcAddr,
                .instance = deviceImpl->instance,
                .physicalDevice = deviceImpl->physicalDevice,
                .device = deviceImpl->device,
                .apiVersion = kVulkanAPIVersion,
                .descriptorHeapEnabled = deviceImpl->bindlessDescriptorHeapEnabled,
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
            deviceImpl->vulkanCapabilities.streamline = true;
            deviceImpl->vulkanCapabilities.streamlineDlssSr = vulkan::streamlineDlssSrSupported();
            deviceImpl->vulkanCapabilities.streamlineDlssRr = vulkan::streamlineDlssRrSupported();
            if (!deviceImpl->vulkanCapabilities.streamlineDlssSr && !streamlineLog.empty()) {
                spdlog::warn("NVIDIA Streamline DLSS-SR unsupported: {}", streamlineLog);
            }
            if (!deviceImpl->vulkanCapabilities.streamlineDlssRr && !streamlineLog.empty()) {
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
        vulkan::toolingHooks().diagnosticsInitialized() &&
        !selectedFeatures.aftermath) {
        spdlog::warn(
            "NVIDIA Nsight Aftermath initialized, but the selected Vulkan device is missing required diagnostics support.");
    }
    if (selectedFeatures.cooperativeVector) {
        spdlog::info("[Vulkan] VK_NV_cooperative_vector enabled");
    }

    return std::unique_ptr<Device>(new Device(std::move(deviceImpl)));
}

namespace detail {

struct VulkanNativeAccess {
    static Result<std::unique_ptr<ComputePipeline>> createMappedComputePipeline(Device& device,
        const ComputePipelineDesc& desc, std::span<const ShaderBindingMappingDesc> mappings)
    {
        return device.createComputePipelineImpl(desc, mappings);
    }

    static QueueImpl* queue(Queue& queue) { return queue.impl_.get(); }
    static DeviceImpl* device(Device& device) { return device.impl_.get(); }
    static Result<std::unique_ptr<TextureView>> retainView(TextureView& view)
    {
        if (!view.impl_) { return makeError(Error::InvalidArgument); }
        auto result = view.impl_->materialize();
        if (!result) { return std::unexpected(result.error()); }
        auto retained = std::make_unique<TextureView>();
        retained->impl_ = view.impl_;
        return retained;
    }
    static vulkan::VulkanDeviceCapabilities deviceCapabilities(const Device& device)
    {
        return device.impl_ ? device.impl_->vulkanCapabilities : vulkan::VulkanDeviceCapabilities{};
    }

    static vulkan::NativeDevice nativeDevice(Device& device)
    {
        if (device.impl_ == nullptr) {
            return {};
        }
        return vulkan::NativeDevice{
            .functions = &device.impl_->functions,
            .instanceFunctions = &device.impl_->instanceFunctions,
            .properties = &device.impl_->physicalProperties,
            .getInstanceProcAddr = device.impl_->getInstanceProcAddr,
            .instance = device.impl_->instance,
            .physicalDevice = device.impl_->physicalDevice,
            .device = device.impl_->device,
            .apiVersion = kVulkanAPIVersion,
            .descriptorHeapEnabled = device.impl_->bindlessDescriptorHeapEnabled,
            .shaderUntypedPointersEnabled = device.impl_->shaderUntypedPointersEnabled,
            .validationEnabled = device.impl_->validationEnabled,
            .synchronizationValidationEnabled = device.impl_->synchronizationValidationEnabled,
            .validationMessengerActive = device.impl_->debugMessenger != VK_NULL_HANDLE,
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

        return vulkan::NativeBuffer{
            .buffer = buffer.impl_->buffer,
            .address = buffer.impl_->address,
            .size = buffer.impl_->desc.size,
            .device = buffer.impl_->device->device,
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
            .mipCount = desc.mipCount,
            .layerCount = desc.layerCount,
            .flags = texture.impl_->flags,
            .usage = texture.impl_->usage,
        };
    }

    static VkShaderModule nativeShaderModule(ShaderModule& shader)
    {
        return shader.impl_ ? shader.impl_->module : VK_NULL_HANDLE;
    }

    static Result<> createCachedGraphicsPipeline(PipelineCache& cache, const VkGraphicsPipelineCreateInfo& info,
        uint64_t stateHash, VkPipeline& pipeline)
    {
        auto* impl = cache.impl_.get();
        if (!impl || !impl->device || !impl->pipelineCache || pipeline != VK_NULL_HANDLE) {
            return makeError(Error::InvalidArgument);
        }
        std::lock_guard lock(impl->mutex);
        VkPipeline candidate = VK_NULL_HANDLE;
        const VkResult result = impl->device->functions.vkCreateGraphicsPipelines(
            impl->device->device, impl->pipelineCache, 1, &info, nullptr, &candidate);
        if (result != VK_SUCCESS) {
            if (candidate) { impl->device->functions.vkDestroyPipeline(impl->device->device, candidate, nullptr); }
            return resultFromVk(result);
        }
        impl->recordPsoLocked(stateHash);
        pipeline = candidate;
        return {};
    }

    static vulkan::NativePipeline nativePipeline(ComputePipeline& pipeline)
    {
        return pipeline.impl_ ? vulkan::NativePipeline{pipeline.impl_->device->device,
            pipeline.impl_->pipeline, pipeline.impl_->layout} : vulkan::NativePipeline{};
    }

    static std::vector<uint8_t> nativeComputeSpirv(ComputePipeline& pipeline, bool deviceCode)
    {
        if (!pipeline.impl_) { return {}; }
        return deviceCode ? pipeline.impl_->replayDeviceSpirv : pipeline.impl_->replayInputSpirv;
    }

    static vulkan::NativePipeline nativePipeline(GraphicsPipeline& pipeline)
    {
        return pipeline.impl_ ? vulkan::NativePipeline{pipeline.impl_->device->device,
            pipeline.impl_->pipeline, pipeline.impl_->layout} : vulkan::NativePipeline{};
    }

    static vulkan::NativeGraphicsShaders nativeShaders(GraphicsShaderObjectProgram& program)
    {
        return program.impl_ ? vulkan::NativeGraphicsShaders{program.impl_->device->device,
            program.impl_->vertexShader, program.impl_->fragmentShader} : vulkan::NativeGraphicsShaders{};
    }

    static const VolkDeviceTable& nativeCommandBufferFunctions(CommandBuffer& commands)
    {
        assert(commands.impl_ != nullptr);
        return commands.impl_->device->functions;
    }

    static VkDevice nativeCommandBufferDevice(CommandBuffer& commands)
    {
        return commands.impl_ ? commands.impl_->device->device : VK_NULL_HANDLE;
    }

    static void invalidateExternalState(CommandBuffer& commands)
    {
        if (!commands.impl_) { return; }
        auto& state = *commands.impl_;
        state.currentComputePipeline = VK_NULL_HANDLE;
        state.currentComputePipelineLayout = VK_NULL_HANDLE;
        state.currentGraphicsPipelineLayout = VK_NULL_HANDLE;
        state.currentComputePipelineUsesBindlessHeap = false;
        state.currentGraphicsPipelineUsesBindlessHeap = false;
        state.currentGraphicsShaderObjectUsesBindlessHeap = false;
        state.currentGraphicsShaderObjectBound = false;
        state.currentBindlessHeap = nullptr;
        state.currentBindlessUserData.clear();
        state.hasCurrentViewport = false;
        state.hasCurrentScissor = false;
    }

    static VkCommandBuffer nativeCommandBuffer(CommandBuffer& commandBuffer)
    {
        return commandBuffer.impl_ != nullptr ? commandBuffer.impl_->commandBuffer : VK_NULL_HANDLE;
    }


    static VkFormat nativeSwapchainFormat(Swapchain& swapchain)
    {
        return swapchain.impl_ != nullptr ? swapchain.impl_->vkFormat : VK_FORMAT_UNDEFINED;
    }

    static VkImageLayout nativeImageLayout(TextureView& view, TextureLayout layout)
    {
        return view.impl_ ? imageLayout(layout, view.impl_->device->vulkanCapabilities.unifiedImageLayouts) : VK_IMAGE_LAYOUT_UNDEFINED;
    }

    static bool hasNativeImageView(const TextureView& view)
    {
        if (!view.impl_) { return false; }
        std::lock_guard lock(view.impl_->mutex);
        return view.impl_->view != VK_NULL_HANDLE;
    }

    static VkImageView nativeImageView(TextureView& view)
    {
        return view.impl_ && view.impl_->materialize() ? view.impl_->view : VK_NULL_HANDLE;
    }

};

Result<std::unique_ptr<ComputePipeline>> createMappedComputePipeline(Device& device,
    const ComputePipelineDesc& desc, std::span<const ShaderBindingMappingDesc> mappings)
{
    return VulkanNativeAccess::createMappedComputePipeline(device, desc, mappings);
}

} // namespace detail

namespace vulkan {

VkFormat nativeFormat(Format format)
{
    return toVkFormat(format);
}

Format resourceFormat(VkFormat format)
{
    return fromVkFormat(format);
}

Result<std::unique_ptr<TextureView>> retainInteropView(TextureView& view)
{
    return detail::VulkanNativeAccess::retainView(view);
}

VkResult submitInterop(Queue& queue, std::span<const VkSubmitInfo2> submits, VkFence fence)
{
    const detail::QueueSubmissionAccess access;
    auto* impl = detail::VulkanNativeAccess::queue(queue);
    if (!impl || submits.size() > UINT32_MAX) { return VK_ERROR_UNKNOWN; }
    std::lock_guard lock(*impl->nativeMutex);
    const RHIOperationScope submitScope(RHIOperation::NativeSubmit, impl->familyIndex);
    VkResult result;
    {
        observeRHICommand(RHICommandEvent::SubmitBegin);
        result = impl->device->functions.vkQueueSubmit2(impl->queue, uint32_t(submits.size()), submits.data(), fence);
        for (const auto& submit : submits) {
            emitTrace({.kind = TraceKind::Submit, .device = impl->device->device,
                .queue = impl->queue, .queueFamily = impl->familyIndex, .submit = &submit, .result = result});
        }
    }
    observeRHICommand(RHICommandEvent::SubmitEnd);
    return result;
}

VkResult presentInterop(Queue& queue, const VkPresentInfoKHR& present)
{
    const detail::QueueSubmissionAccess access;
    auto* impl = detail::VulkanNativeAccess::queue(queue);
    if (!impl) { return VK_ERROR_UNKNOWN; }
    std::lock_guard lock(*impl->nativeMutex);
    return impl->device->functions.vkQueuePresentKHR(impl->queue, &present);
}

VkResult waitInterop(Queue& queue)
{
    const detail::QueueSubmissionAccess access;
    auto* impl = detail::VulkanNativeAccess::queue(queue);
    if (!impl) { return VK_ERROR_UNKNOWN; }
    std::lock_guard lock(*impl->nativeMutex);
    return impl->device->functions.vkQueueWaitIdle(impl->queue);
}

VkResult waitInterop(Device& device)
{
    const QueueSubmissionIsolation isolation;
    auto* impl = detail::VulkanNativeAccess::device(device);
    return impl ? impl->functions.vkDeviceWaitIdle(impl->device) : VK_ERROR_UNKNOWN;
}

ExternalCommandScope::ExternalCommandScope(CommandBuffer& commands)
    : commands_(commands.recording() ? &commands : nullptr)
{
}

ExternalCommandScope::~ExternalCommandScope()
{
    if (commands_) { detail::VulkanNativeAccess::invalidateExternalState(*commands_); }
}

VkCommandBuffer ExternalCommandScope::commandBuffer() const
{
    return commands_ ? detail::VulkanNativeAccess::nativeCommandBuffer(*commands_) : VK_NULL_HANDLE;
}

const VolkDeviceTable& ExternalCommandScope::functions() const
{
    assert(commands_ != nullptr);
    return detail::VulkanNativeAccess::nativeCommandBufferFunctions(*commands_);
}

Result<VkImageView> ExternalCommandScope::imageView(TextureView& view) const
{
    if (!commands_) { return makeError(Error::InvalidArgument); }
    auto retained = commands_->useNativeTextureView(view);
    if (!retained) { return makeError(retained.error()); }
    return detail::VulkanNativeAccess::nativeImageView(view);
}

VulkanDeviceCapabilities deviceCapabilities(const Device& device)
{
    return detail::VulkanNativeAccess::deviceCapabilities(device);
}

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

VkShaderModule nativeShaderModule(ShaderModule& shader)
{
    return detail::VulkanNativeAccess::nativeShaderModule(shader);
}

Result<> createCachedGraphicsPipeline(PipelineCache& cache, const VkGraphicsPipelineCreateInfo& info,
    uint64_t stateHash, VkPipeline& pipeline)
{
    return detail::VulkanNativeAccess::createCachedGraphicsPipeline(cache, info, stateHash, pipeline);
}

NativePipeline nativePipeline(ComputePipeline& pipeline)
{
    return detail::VulkanNativeAccess::nativePipeline(pipeline);
}

std::vector<uint8_t> nativeComputeSpirv(ComputePipeline& pipeline, bool deviceCode)
{
    return detail::VulkanNativeAccess::nativeComputeSpirv(pipeline, deviceCode);
}

NativePipeline nativePipeline(GraphicsPipeline& pipeline)
{
    return detail::VulkanNativeAccess::nativePipeline(pipeline);
}

NativeGraphicsShaders nativeShaders(GraphicsShaderObjectProgram& program)
{
    return detail::VulkanNativeAccess::nativeShaders(program);
}

VkDevice nativeCommandBufferDevice(CommandBuffer& commands)
{
    return detail::VulkanNativeAccess::nativeCommandBufferDevice(commands);
}

VkFormat nativeSwapchainFormat(Swapchain& swapchain)
{
    return detail::VulkanNativeAccess::nativeSwapchainFormat(swapchain);
}

VkImageLayout nativeImageLayout(TextureView& view, TextureLayout layout)
{
    return detail::VulkanNativeAccess::nativeImageLayout(view, layout);
}

bool hasNativeImageView(const TextureView& view)
{
    return detail::VulkanNativeAccess::hasNativeImageView(view);
}

VkImageView nativeImageView(TextureView& view)
{
    return detail::VulkanNativeAccess::nativeImageView(view);
}

} // namespace vulkan


} // namespace metallic::render
