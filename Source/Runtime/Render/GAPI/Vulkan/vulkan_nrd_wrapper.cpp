#include "Runtime/Render/GAPI/Vulkan/vulkan_nrd_wrapper.h"

#include "Runtime/Render/GAPI/Vulkan/vulkan_native.h"

#include <volk.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
#include <string_view>

namespace metallic::render::vulkan {
namespace {

#if METALLIC_HAS_NRD

constexpr uint32_t kNrdSamplerSetIndex = 1;
constexpr uint32_t kNrdResourceSetIndex = 0;

Result resultFromNrd(nrd::Result result)
{
    switch (result) {
    case nrd::Result::SUCCESS:
        return {};
    case nrd::Result::INVALID_ARGUMENT:
    case nrd::Result::NON_UNIQUE_IDENTIFIER:
        return makeError(Error::InvalidArgument);
    case nrd::Result::UNSUPPORTED:
        return makeError(Error::Unsupported);
    case nrd::Result::FAILURE:
    case nrd::Result::MAX_NUM:
        return makeError(Error::Failure);
    }
    return makeError(Error::Failure);
}

std::string nrdResultMessage(std::string_view label, nrd::Result result)
{
    std::string message(label);
    message += " returned ";
    message += std::to_string(static_cast<uint32_t>(result));
    return message;
}

Result resultFromVk(VkResult result)
{
    if (result == VK_SUCCESS) {
        return {};
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
        return makeError(Error::OutOfMemory);
    }
    if (result == VK_ERROR_DEVICE_LOST) {
        return makeError(Error::DeviceLost);
    }
    return makeError(Error::Failure);
}

std::string vkResultMessage(std::string_view label, VkResult result)
{
    std::string message(label);
    message += " returned ";
    message += std::to_string(static_cast<int32_t>(result));
    return message;
}

uint16_t divideRoundUp(uint32_t dividend, uint16_t divisor)
{
    return static_cast<uint16_t>((dividend + divisor - 1u) / divisor);
}

Format formatFromNrd(nrd::Format format)
{
    switch (format) {
    case nrd::Format::R8_UNORM:
        return Format::R8Unorm;
    case nrd::Format::R8_SNORM:
        return Format::R8Snorm;
    case nrd::Format::R8_UINT:
        return Format::R8Uint;
    case nrd::Format::R8_SINT:
        return Format::R8Sint;
    case nrd::Format::RG8_UNORM:
        return Format::Rg8Unorm;
    case nrd::Format::RG8_SNORM:
        return Format::Rg8Snorm;
    case nrd::Format::RG8_UINT:
        return Format::Rg8Uint;
    case nrd::Format::RG8_SINT:
        return Format::Rg8Sint;
    case nrd::Format::RGBA8_UNORM:
        return Format::Rgba8Unorm;
    case nrd::Format::RGBA8_SNORM:
        return Format::Rgba8Snorm;
    case nrd::Format::RGBA8_UINT:
        return Format::Rgba8Uint;
    case nrd::Format::RGBA8_SINT:
        return Format::Rgba8Sint;
    case nrd::Format::RGBA8_SRGB:
        return Format::Rgba8Srgb;
    case nrd::Format::R16_UNORM:
        return Format::R16Unorm;
    case nrd::Format::R16_SNORM:
        return Format::R16Snorm;
    case nrd::Format::R16_UINT:
        return Format::R16Uint;
    case nrd::Format::R16_SINT:
        return Format::R16Sint;
    case nrd::Format::R16_SFLOAT:
        return Format::R16Sfloat;
    case nrd::Format::RG16_UNORM:
        return Format::Rg16Unorm;
    case nrd::Format::RG16_SNORM:
        return Format::Rg16Snorm;
    case nrd::Format::RG16_UINT:
        return Format::Rg16Uint;
    case nrd::Format::RG16_SINT:
        return Format::Rg16Sint;
    case nrd::Format::RG16_SFLOAT:
        return Format::Rg16Sfloat;
    case nrd::Format::RGBA16_UNORM:
        return Format::Rgba16Unorm;
    case nrd::Format::RGBA16_SNORM:
        return Format::Rgba16Snorm;
    case nrd::Format::RGBA16_UINT:
        return Format::Rgba16Uint;
    case nrd::Format::RGBA16_SINT:
        return Format::Rgba16Sint;
    case nrd::Format::RGBA16_SFLOAT:
        return Format::Rgba16Sfloat;
    case nrd::Format::R32_UINT:
        return Format::R32Uint;
    case nrd::Format::R32_SINT:
        return Format::R32Sint;
    case nrd::Format::R32_SFLOAT:
        return Format::R32Sfloat;
    case nrd::Format::RG32_UINT:
        return Format::Rg32Uint;
    case nrd::Format::RG32_SINT:
        return Format::Rg32Sint;
    case nrd::Format::RG32_SFLOAT:
        return Format::Rg32Sfloat;
    case nrd::Format::RGB32_UINT:
        return Format::Rgb32Uint;
    case nrd::Format::RGB32_SINT:
        return Format::Rgb32Sint;
    case nrd::Format::RGB32_SFLOAT:
        return Format::Rgb32Sfloat;
    case nrd::Format::RGBA32_UINT:
        return Format::Rgba32Uint;
    case nrd::Format::RGBA32_SINT:
        return Format::Rgba32Sint;
    case nrd::Format::RGBA32_SFLOAT:
        return Format::Rgba32Sfloat;
    case nrd::Format::R10_G10_B10_A2_UNORM:
        return Format::A2B10G10R10UnormPack32;
    case nrd::Format::R10_G10_B10_A2_UINT:
        return Format::A2R10G10B10UintPack32;
    case nrd::Format::R11_G11_B10_UFLOAT:
        return Format::B10G11R11UfloatPack32;
    case nrd::Format::R9_G9_B9_E5_UFLOAT:
        return Format::E5B9G9R9UfloatPack32;
    case nrd::Format::MAX_NUM:
        return Format::Unknown;
    }
    return Format::Unknown;
}

VkDescriptorType descriptorTypeFromNrd(nrd::DescriptorType type)
{
    switch (type) {
    case nrd::DescriptorType::TEXTURE:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case nrd::DescriptorType::STORAGE_TEXTURE:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case nrd::DescriptorType::MAX_NUM:
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

VkFilter filterFromNrd(nrd::Sampler sampler)
{
    switch (sampler) {
    case nrd::Sampler::NEAREST_CLAMP:
        return VK_FILTER_NEAREST;
    case nrd::Sampler::LINEAR_CLAMP:
        return VK_FILTER_LINEAR;
    case nrd::Sampler::MAX_NUM:
        return VK_FILTER_NEAREST;
    }
    return VK_FILTER_NEAREST;
}

VkImageMemoryBarrier imageBarrier(VkImage image, VkAccessFlags srcAccess, VkAccessFlags dstAccess)
{
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = srcAccess,
        .dstAccessMask = dstAccess,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
}

#endif

} // namespace

Format nrdNormalRoughnessFormat()
{
#if METALLIC_HAS_NRD
    switch (nrd::GetLibraryDesc()->normalEncoding) {
    case nrd::NormalEncoding::RGBA8_UNORM:
        return Format::Rgba8Unorm;
    case nrd::NormalEncoding::RGBA8_SNORM:
        return Format::Rgba8Snorm;
    case nrd::NormalEncoding::R10_G10_B10_A2_UNORM:
        return Format::A2B10G10R10UnormPack32;
    case nrd::NormalEncoding::RGBA16_UNORM:
        return Format::Rgba16Unorm;
    case nrd::NormalEncoding::RGBA16_SNORM:
        return Format::Rgba16Snorm;
    case nrd::NormalEncoding::MAX_NUM:
        return Format::Unknown;
    }
#endif
    return Format::A2B10G10R10UnormPack32;
}

#if METALLIC_HAS_NRD

struct NrdDenoiser::Impl {
    nrd::Instance* instance = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    uint16_t width = 0;
    uint16_t height = 0;
    std::vector<TextureResource> permanentTextures;
    std::vector<TextureResource> transientTextures;
    NrdUserTexturePool userTexturePool{};
    std::vector<VkSampler> samplers;
    std::unique_ptr<Buffer> constantBuffer;
    std::vector<Pipeline> pipelines;
    VkDescriptorSetLayout samplerDescriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool samplerDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet samplerDescriptorSet = VK_NULL_HANDLE;
};

NrdDenoiser::NrdDenoiser()
    : impl_(std::make_unique<Impl>())
{
}

NrdDenoiser::~NrdDenoiser()
{
    clear();
}

NrdDenoiser::NrdDenoiser(NrdDenoiser&&) noexcept = default;
NrdDenoiser& NrdDenoiser::operator=(NrdDenoiser&&) noexcept = default;

Result NrdDenoiser::initialize(
    Device& device,
    Queue& queue,
    uint16_t width,
    uint16_t height,
    const NrdUserTexturePool& userTexturePool,
    std::string& log)
{
    if (width == 0 || height == 0) {
        log = "NrdDenoiser requires a non-zero image size";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().pushDescriptor) {
        log = "NrdDenoiser requires VK_KHR_push_descriptor";
        return makeError(Error::Unsupported);
    }

    const NativeDevice nativeDeviceInfo = nativeDevice(device);
    if (nativeDeviceInfo.device == VK_NULL_HANDLE) {
        log = "NrdDenoiser requires a Vulkan device";
        return makeError(Error::InvalidArgument);
    }
    volkLoadDevice(nativeDeviceInfo.device);

    clear();
    impl_ = std::make_unique<Impl>();
    impl_->device = nativeDeviceInfo.device;
    impl_->width = width;
    impl_->height = height;
    impl_->userTexturePool = userTexturePool;

    const std::array<nrd::DenoiserDesc, 4> denoisers = {
        nrd::DenoiserDesc{
            .identifier = static_cast<nrd::Identifier>(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR),
            .denoiser = nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR,
        },
        nrd::DenoiserDesc{
            .identifier = static_cast<nrd::Identifier>(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR),
            .denoiser = nrd::Denoiser::RELAX_DIFFUSE_SPECULAR,
        },
        nrd::DenoiserDesc{
            .identifier = static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE),
            .denoiser = nrd::Denoiser::REFERENCE,
        },
        nrd::DenoiserDesc{
            .identifier = static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE) + 1,
            .denoiser = nrd::Denoiser::REFERENCE,
        },
    };
    nrd::InstanceCreationDesc instanceDesc{
        .denoisers = denoisers.data(),
        .denoisersNum = static_cast<uint32_t>(denoisers.size()),
    };
    nrd::Result nrdResult = nrd::CreateInstance(instanceDesc, impl_->instance);
    if (nrdResult != nrd::Result::SUCCESS) {
        log = nrdResultMessage("nrd::CreateInstance", nrdResult);
        clear();
        return resultFromNrd(nrdResult);
    }

    const nrd::InstanceDesc* instanceDescInfo = nrd::GetInstanceDesc(*impl_->instance);
    if (instanceDescInfo == nullptr ||
        instanceDescInfo->constantBufferAndSamplersSpaceIndex != kNrdSamplerSetIndex ||
        instanceDescInfo->resourcesSpaceIndex != kNrdResourceSetIndex) {
        log = "NrdDenoiser got an unsupported NRD descriptor space layout";
        clear();
        return makeError(Error::Unsupported);
    }

    impl_->permanentTextures.resize(instanceDescInfo->permanentPoolSize);
    for (uint32_t index = 0; index < instanceDescInfo->permanentPoolSize; ++index) {
        Result result = createInternalTexture(
            device,
            instanceDescInfo->permanentPool[index],
            width,
            height,
            impl_->permanentTextures[index],
            log);
        if (!result) {
            clear();
            return result;
        }
    }

    impl_->transientTextures.resize(instanceDescInfo->transientPoolSize);
    for (uint32_t index = 0; index < instanceDescInfo->transientPoolSize; ++index) {
        Result result = createInternalTexture(
            device,
            instanceDescInfo->transientPool[index],
            width,
            height,
            impl_->transientTextures[index],
            log);
        if (!result) {
            clear();
            return result;
        }
    }

    Result result = initializeInternalTextureLayouts(device, queue, log);
    if (!result) {
        clear();
        return result;
    }

    impl_->samplers.reserve(instanceDescInfo->samplersNum);
    for (uint32_t samplerIndex = 0; samplerIndex < instanceDescInfo->samplersNum; ++samplerIndex) {
        const VkFilter filter = filterFromNrd(instanceDescInfo->samplers[samplerIndex]);
        VkSamplerCreateInfo samplerInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = filter,
            .minFilter = filter,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod = VK_LOD_CLAMP_NONE,
        };
        VkSampler sampler = VK_NULL_HANDLE;
        const VkResult vkResult = vkCreateSampler(impl_->device, &samplerInfo, nullptr, &sampler);
        if (vkResult != VK_SUCCESS) {
            log = vkResultMessage("vkCreateSampler(NRD)", vkResult);
            clear();
            return resultFromVk(vkResult);
        }
        impl_->samplers.push_back(sampler);
    }

    result = device.createBuffer(
        BufferDesc{
            .size = instanceDescInfo->constantBufferMaxDataSize,
            .usage = BufferUsageBits::Constant | BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::Device,
        },
        impl_->constantBuffer);
    if (!result || impl_->constantBuffer == nullptr) {
        log = "createBuffer(NRD constant buffer) returned ";
        log += resultToString(result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    result = createPipelines(log);
    if (!result) {
        clear();
        return result;
    }
    return {};
}

void NrdDenoiser::clear()
{
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(impl_->device);
        for (Pipeline& pipeline : impl_->pipelines) {
            if (pipeline.pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(impl_->device, pipeline.pipeline, nullptr);
            }
            if (pipeline.pipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(impl_->device, pipeline.pipelineLayout, nullptr);
            }
            if (pipeline.resourceDescriptorLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(
                    impl_->device,
                    pipeline.resourceDescriptorLayout,
                    nullptr);
            }
        }
        if (impl_->samplerDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(impl_->device, impl_->samplerDescriptorPool, nullptr);
        }
        if (impl_->samplerDescriptorLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(impl_->device, impl_->samplerDescriptorLayout, nullptr);
        }
        for (VkSampler sampler : impl_->samplers) {
            vkDestroySampler(impl_->device, sampler, nullptr);
        }
    }
    if (impl_->instance != nullptr) {
        nrd::DestroyInstance(*impl_->instance);
        impl_->instance = nullptr;
    }
    impl_->permanentTextures.clear();
    impl_->transientTextures.clear();
    impl_->constantBuffer.reset();
    impl_->samplers.clear();
    impl_->pipelines.clear();
    impl_->samplerDescriptorLayout = VK_NULL_HANDLE;
    impl_->samplerDescriptorPool = VK_NULL_HANDLE;
    impl_->samplerDescriptorSet = VK_NULL_HANDLE;
    impl_->device = VK_NULL_HANDLE;
    impl_->width = 0;
    impl_->height = 0;
}

bool NrdDenoiser::valid() const
{
    return impl_ != nullptr &&
        impl_->instance != nullptr &&
        impl_->device != VK_NULL_HANDLE &&
        impl_->constantBuffer != nullptr &&
        !impl_->pipelines.empty();
}

uint16_t NrdDenoiser::width() const
{
    return impl_ != nullptr ? impl_->width : 0;
}

uint16_t NrdDenoiser::height() const
{
    return impl_ != nullptr ? impl_->height : 0;
}

void NrdDenoiser::setUserPoolTexture(nrd::ResourceType resource, Texture& texture, TextureView& view)
{
    if (impl_ == nullptr || static_cast<size_t>(resource) >= impl_->userTexturePool.size()) {
        return;
    }
    impl_->userTexturePool[static_cast<size_t>(resource)] = NrdTextureRef{
        .texture = &texture,
        .view = &view,
    };
}

Result NrdDenoiser::setCommonSettings(const nrd::CommonSettings& settings)
{
    if (!valid()) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromNrd(nrd::SetCommonSettings(*impl_->instance, settings));
}

Result NrdDenoiser::setReblurSettings(const nrd::ReblurSettings& settings)
{
    return setDenoiserSettings(
        static_cast<nrd::Identifier>(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR),
        &settings);
}

Result NrdDenoiser::setRelaxSettings(const nrd::RelaxSettings& settings)
{
    return setDenoiserSettings(
        static_cast<nrd::Identifier>(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR),
        &settings);
}

Result NrdDenoiser::denoise(NrdDenoiserMode mode, CommandBuffer& commandBuffer)
{
    if (mode == NrdDenoiserMode::Reference) {
        const std::array<nrd::Identifier, 2> identifiers = {
            static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE),
            static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE) + 1,
        };
        return denoiseIdentifiers(identifiers.data(), static_cast<uint32_t>(identifiers.size()), commandBuffer);
    }

    const nrd::Identifier identifier = mode == NrdDenoiserMode::Relax
        ? static_cast<nrd::Identifier>(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR)
        : static_cast<nrd::Identifier>(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR);
    return denoiseIdentifiers(&identifier, 1, commandBuffer);
}

Result NrdDenoiser::denoiseIdentifiers(
    const nrd::Identifier* denoisers,
    uint32_t denoiserCount,
    CommandBuffer& commandBuffer)
{
    if (!valid() || denoisers == nullptr || denoiserCount == 0) {
        return makeError(Error::InvalidArgument);
    }

    const nrd::DispatchDesc* dispatches = nullptr;
    uint32_t dispatchCount = 0;
    nrd::Result nrdResult = nrd::GetComputeDispatches(
        *impl_->instance,
        denoisers,
        denoiserCount,
        dispatches,
        dispatchCount);
    if (nrdResult != nrd::Result::SUCCESS) {
        return resultFromNrd(nrdResult);
    }
    for (uint32_t dispatchIndex = 0; dispatchIndex < dispatchCount; ++dispatchIndex) {
        Result result = dispatch(commandBuffer, dispatches[dispatchIndex]);
        if (!result) {
            return result;
        }
    }
    return {};
}

Result NrdDenoiser::createInternalTexture(
    Device& device,
    const nrd::TextureDesc& desc,
    uint16_t width,
    uint16_t height,
    TextureResource& outResource,
    std::string& log)
{
    const Format format = formatFromNrd(desc.format);
    if (format == Format::Unknown) {
        log = "NrdDenoiser received an unsupported texture format";
        return makeError(Error::Unsupported);
    }

    const uint16_t textureWidth = divideRoundUp(width, desc.downsampleFactor);
    const uint16_t textureHeight = divideRoundUp(height, desc.downsampleFactor);
    Result result = device.createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage,
            .format = format,
            .width = textureWidth,
            .height = textureHeight,
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        },
        outResource.texture);
    if (!result || outResource.texture == nullptr) {
        log = "createTexture(NRD internal) returned ";
        log += resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }
    result = device.createTextureView(
        *outResource.texture,
        TextureViewDesc{
            .format = format,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        },
        outResource.view);
    if (!result || outResource.view == nullptr) {
        log = "createTextureView(NRD internal) returned ";
        log += resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

Result NrdDenoiser::initializeInternalTextureLayouts(Device& device, Queue& queue, std::string& log)
{
    std::unique_ptr<CommandPool> commandPool;
    Result result = device.createCommandPool(queue, commandPool);
    if (!result || commandPool == nullptr) {
        log = "createCommandPool(NRD init) returned ";
        log += resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }
    std::unique_ptr<CommandBuffer> commandBuffer;
    result = commandPool->createCommandBuffer(commandBuffer);
    if (!result || commandBuffer == nullptr) {
        log = "createCommandBuffer(NRD init) returned ";
        log += resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }
    std::unique_ptr<Fence> fence;
    result = device.createFence(false, fence);
    if (!result || fence == nullptr) {
        log = "createFence(NRD init) returned ";
        log += resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }

    result = commandBuffer->begin();
    if (!result) {
        log = "CommandBuffer::begin(NRD init) returned ";
        log += resultToString(result);
        return result;
    }

    std::vector<TextureBarrierDesc> barriers;
    barriers.reserve(impl_->permanentTextures.size() + impl_->transientTextures.size());
    auto appendBarrier = [&barriers](TextureResource& resource) {
        if (resource.texture != nullptr) {
            barriers.push_back(TextureBarrierDesc{
                .texture = resource.texture.get(),
                .before = ResourceState::Undefined,
                .after = ResourceState::General,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            });
        }
    };
    for (TextureResource& resource : impl_->permanentTextures) {
        appendBarrier(resource);
    }
    for (TextureResource& resource : impl_->transientTextures) {
        appendBarrier(resource);
    }
    if (!barriers.empty()) {
        commandBuffer->barrier(BarrierDesc{
            .textures = barriers.data(),
            .textureCount = static_cast<uint32_t>(barriers.size()),
        });
    }

    VkCommandBuffer vkCommandBuffer = nativeCommandBuffer(*commandBuffer);
    const VkClearColorValue clear = {{0.0f, 0.0f, 0.0f, 0.0f}};
    const VkImageSubresourceRange range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    auto clearResource = [&](TextureResource& resource) {
        if (resource.texture != nullptr) {
            const NativeTexture native = nativeTexture(*resource.texture);
            if (native.image != VK_NULL_HANDLE) {
                vkCmdClearColorImage(vkCommandBuffer, native.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            }
        }
    };
    for (TextureResource& resource : impl_->permanentTextures) {
        clearResource(resource);
    }
    for (TextureResource& resource : impl_->transientTextures) {
        clearResource(resource);
    }

    result = commandBuffer->end();
    if (!result) {
        log = "CommandBuffer::end(NRD init) returned ";
        log += resultToString(result);
        return result;
    }
    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
    result = queue.submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = fence.get(),
    });
    if (!result) {
        log = "Queue::submit(NRD init) returned ";
        log += resultToString(result);
        return result;
    }
    result = fence->wait();
    if (!result) {
        log = "Fence::wait(NRD init) returned ";
        log += resultToString(result);
    }
    return result;
}

Result NrdDenoiser::createPipelines(std::string& log)
{
    const nrd::InstanceDesc* instanceDesc = nrd::GetInstanceDesc(*impl_->instance);
    const nrd::LibraryDesc* libraryDesc = nrd::GetLibraryDesc();
    if (instanceDesc == nullptr || libraryDesc == nullptr) {
        log = "NrdDenoiser failed to get NRD descriptors";
        return makeError(Error::Failure);
    }

    const uint32_t constantBufferBinding = libraryDesc->spirvBindingOffsets.constantBufferOffset;
    const uint32_t samplerBindingOffset = libraryDesc->spirvBindingOffsets.samplerOffset;
    const uint32_t textureBindingOffset = libraryDesc->spirvBindingOffsets.textureOffset;
    const uint32_t storageBindingOffset = libraryDesc->spirvBindingOffsets.storageTextureAndBufferOffset;

    std::vector<VkDescriptorSetLayoutBinding> samplerBindings;
    samplerBindings.reserve(instanceDesc->samplersNum + 1);
    for (uint32_t samplerIndex = 0; samplerIndex < instanceDesc->samplersNum; ++samplerIndex) {
        samplerBindings.push_back(VkDescriptorSetLayoutBinding{
            .binding = samplerBindingOffset + samplerIndex,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = &impl_->samplers[samplerIndex],
        });
    }
    samplerBindings.push_back(VkDescriptorSetLayoutBinding{
        .binding = constantBufferBinding,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    });

    VkDescriptorSetLayoutCreateInfo samplerLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(samplerBindings.size()),
        .pBindings = samplerBindings.data(),
    };
    VkResult vkResult = vkCreateDescriptorSetLayout(
        impl_->device,
        &samplerLayoutInfo,
        nullptr,
        &impl_->samplerDescriptorLayout);
    if (vkResult != VK_SUCCESS) {
        log = vkResultMessage("vkCreateDescriptorSetLayout(NRD samplers)", vkResult);
        return resultFromVk(vkResult);
    }

    std::array<VkDescriptorPoolSize, 2> poolSizes = {
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = instanceDesc->samplersNum,
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
        },
    };
    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    vkResult = vkCreateDescriptorPool(impl_->device, &poolInfo, nullptr, &impl_->samplerDescriptorPool);
    if (vkResult != VK_SUCCESS) {
        log = vkResultMessage("vkCreateDescriptorPool(NRD samplers)", vkResult);
        return resultFromVk(vkResult);
    }

    VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = impl_->samplerDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &impl_->samplerDescriptorLayout,
    };
    vkResult = vkAllocateDescriptorSets(impl_->device, &allocateInfo, &impl_->samplerDescriptorSet);
    if (vkResult != VK_SUCCESS) {
        log = vkResultMessage("vkAllocateDescriptorSets(NRD samplers)", vkResult);
        return resultFromVk(vkResult);
    }

    const NativeBuffer constantBuffer = nativeBuffer(*impl_->constantBuffer);
    VkDescriptorBufferInfo constantBufferInfo{
        .buffer = constantBuffer.buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkWriteDescriptorSet constantBufferWrite{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = impl_->samplerDescriptorSet,
        .dstBinding = constantBufferBinding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &constantBufferInfo,
    };
    vkUpdateDescriptorSets(impl_->device, 1, &constantBufferWrite, 0, nullptr);

    uint32_t maxResourceBindings = 0;
    for (uint32_t pipelineIndex = 0; pipelineIndex < instanceDesc->pipelinesNum; ++pipelineIndex) {
        const nrd::PipelineDesc& pipelineDesc = instanceDesc->pipelines[pipelineIndex];
        uint32_t bindingCount = 0;
        for (uint32_t rangeIndex = 0; rangeIndex < pipelineDesc.resourceRangesNum; ++rangeIndex) {
            bindingCount += pipelineDesc.resourceRanges[rangeIndex].descriptorsNum;
        }
        maxResourceBindings = std::max(maxResourceBindings, bindingCount);
    }

    impl_->pipelines.resize(instanceDesc->pipelinesNum);
    for (uint32_t pipelineIndex = 0; pipelineIndex < instanceDesc->pipelinesNum; ++pipelineIndex) {
        const nrd::PipelineDesc& pipelineDesc = instanceDesc->pipelines[pipelineIndex];
        if (pipelineDesc.computeShaderSPIRV.bytecode == nullptr || pipelineDesc.computeShaderSPIRV.size == 0) {
            log = "NrdDenoiser requires NRD SPIR-V shader blobs";
            return makeError(Error::Unsupported);
        }

        std::vector<VkDescriptorSetLayoutBinding> resourceBindings;
        resourceBindings.reserve(maxResourceBindings);
        for (uint32_t rangeIndex = 0; rangeIndex < pipelineDesc.resourceRangesNum; ++rangeIndex) {
            const nrd::ResourceRangeDesc& range = pipelineDesc.resourceRanges[rangeIndex];
            const bool isStorage = range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
            const VkDescriptorType descriptorType = descriptorTypeFromNrd(range.descriptorType);
            const uint32_t baseBinding = isStorage ? storageBindingOffset : textureBindingOffset;
            for (uint32_t descriptorIndex = 0; descriptorIndex < range.descriptorsNum; ++descriptorIndex) {
                resourceBindings.push_back(VkDescriptorSetLayoutBinding{
                    .binding = baseBinding + instanceDesc->resourcesBaseRegisterIndex + descriptorIndex,
                    .descriptorType = descriptorType,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                });
            }
        }

        Pipeline& pipeline = impl_->pipelines[pipelineIndex];
        pipeline.bindingCount = static_cast<uint32_t>(resourceBindings.size());
        VkDescriptorSetLayoutCreateInfo resourceLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
            .bindingCount = static_cast<uint32_t>(resourceBindings.size()),
            .pBindings = resourceBindings.data(),
        };
        VkDescriptorSetLayout resourceLayout = VK_NULL_HANDLE;
        vkResult = vkCreateDescriptorSetLayout(impl_->device, &resourceLayoutInfo, nullptr, &resourceLayout);
        if (vkResult != VK_SUCCESS) {
            log = vkResultMessage("vkCreateDescriptorSetLayout(NRD resources)", vkResult);
            return resultFromVk(vkResult);
        }
        pipeline.resourceDescriptorLayout = resourceLayout;

        const std::array<VkDescriptorSetLayout, 2> setLayouts = {
            resourceLayout,
            impl_->samplerDescriptorLayout,
        };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts = setLayouts.data(),
        };
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        vkResult = vkCreatePipelineLayout(impl_->device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
        if (vkResult != VK_SUCCESS) {
            log = vkResultMessage("vkCreatePipelineLayout(NRD)", vkResult);
            return resultFromVk(vkResult);
        }
        pipeline.pipelineLayout = pipelineLayout;

        VkShaderModuleCreateInfo shaderInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = static_cast<size_t>(pipelineDesc.computeShaderSPIRV.size),
            .pCode = static_cast<const uint32_t*>(pipelineDesc.computeShaderSPIRV.bytecode),
        };
        VkShaderModule shaderModule = VK_NULL_HANDLE;
        vkResult = vkCreateShaderModule(impl_->device, &shaderInfo, nullptr, &shaderModule);
        if (vkResult != VK_SUCCESS) {
            log = vkResultMessage("vkCreateShaderModule(NRD)", vkResult);
            return resultFromVk(vkResult);
        }

        VkPipelineShaderStageCreateInfo stageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shaderModule,
            .pName = instanceDesc->shaderEntryPoint != nullptr ? instanceDesc->shaderEntryPoint : "main",
        };
        VkComputePipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stageInfo,
            .layout = pipelineLayout,
        };
        VkPipeline vkPipeline = VK_NULL_HANDLE;
        vkResult = vkCreateComputePipelines(
            impl_->device,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &vkPipeline);
        vkDestroyShaderModule(impl_->device, shaderModule, nullptr);
        if (vkResult != VK_SUCCESS) {
            log = vkResultMessage("vkCreateComputePipelines(NRD)", vkResult);
            return resultFromVk(vkResult);
        }
        pipeline.pipeline = vkPipeline;
    }
    return {};
}

Result NrdDenoiser::setDenoiserSettings(nrd::Identifier identifier, const void* settings)
{
    if (!valid() || settings == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromNrd(nrd::SetDenoiserSettings(*impl_->instance, identifier, settings));
}

Result NrdDenoiser::dispatch(CommandBuffer& commandBuffer, const nrd::DispatchDesc& dispatchDesc)
{
    if (!valid() || dispatchDesc.pipelineIndex >= impl_->pipelines.size()) {
        return makeError(Error::InvalidArgument);
    }

    VkCommandBuffer vkCommandBuffer = nativeCommandBuffer(commandBuffer);
    if (vkCommandBuffer == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }

    const nrd::InstanceDesc* instanceDesc = nrd::GetInstanceDesc(*impl_->instance);
    const nrd::LibraryDesc* libraryDesc = nrd::GetLibraryDesc();
    const nrd::PipelineDesc& pipelineDesc = instanceDesc->pipelines[dispatchDesc.pipelineIndex];
    const uint32_t textureBindingOffset = libraryDesc->spirvBindingOffsets.textureOffset;
    const uint32_t storageBindingOffset = libraryDesc->spirvBindingOffsets.storageTextureAndBufferOffset;
    Pipeline& pipeline = impl_->pipelines[dispatchDesc.pipelineIndex];

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkImageMemoryBarrier> imageBarriers;
    writes.reserve(pipeline.bindingCount);
    imageInfos.reserve(pipeline.bindingCount);
    imageBarriers.reserve(pipeline.bindingCount);

    uint32_t resourceIndex = 0;
    for (uint32_t rangeIndex = 0; rangeIndex < pipelineDesc.resourceRangesNum; ++rangeIndex) {
        const nrd::ResourceRangeDesc& range = pipelineDesc.resourceRanges[rangeIndex];
        const bool isStorage = range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
        const uint32_t baseBinding = isStorage ? storageBindingOffset : textureBindingOffset;
        for (uint32_t descriptorIndex = 0; descriptorIndex < range.descriptorsNum; ++descriptorIndex) {
            if (resourceIndex >= dispatchDesc.resourcesNum) {
                return makeError(Error::InvalidArgument);
            }
            const nrd::ResourceDesc& resourceDesc = dispatchDesc.resources[resourceIndex];
            NrdTextureRef resource;
            if (resourceDesc.type == nrd::ResourceType::TRANSIENT_POOL) {
                if (resourceDesc.indexInPool >= impl_->transientTextures.size()) {
                    return makeError(Error::InvalidArgument);
                }
                TextureResource& textureResource = impl_->transientTextures[resourceDesc.indexInPool];
                resource.texture = textureResource.texture.get();
                resource.view = textureResource.view.get();
            } else if (resourceDesc.type == nrd::ResourceType::PERMANENT_POOL) {
                if (resourceDesc.indexInPool >= impl_->permanentTextures.size()) {
                    return makeError(Error::InvalidArgument);
                }
                TextureResource& textureResource = impl_->permanentTextures[resourceDesc.indexInPool];
                resource.texture = textureResource.texture.get();
                resource.view = textureResource.view.get();
            } else {
                const size_t userResourceIndex = static_cast<size_t>(resourceDesc.type);
                if (userResourceIndex >= impl_->userTexturePool.size()) {
                    return makeError(Error::InvalidArgument);
                }
                resource = impl_->userTexturePool[userResourceIndex];
            }

            if (resource.texture == nullptr || resource.view == nullptr) {
                return makeError(Error::InvalidArgument);
            }

            const NativeTexture nativeTextureInfo = nativeTexture(*resource.texture);
            const VkImageView imageView = nativeImageView(*resource.view);
            if (nativeTextureInfo.image == VK_NULL_HANDLE || imageView == VK_NULL_HANDLE) {
                return makeError(Error::InvalidArgument);
            }

            imageInfos.push_back(VkDescriptorImageInfo{
                .imageView = imageView,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            });
            writes.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = baseBinding + instanceDesc->resourcesBaseRegisterIndex + descriptorIndex,
                .descriptorCount = 1,
                .descriptorType = descriptorTypeFromNrd(resourceDesc.descriptorType),
                .pImageInfo = &imageInfos.back(),
            });
            imageBarriers.push_back(isStorage
                ? imageBarrier(nativeTextureInfo.image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
                : imageBarrier(nativeTextureInfo.image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
            ++resourceIndex;
        }
    }

    if (!imageBarriers.empty()) {
        vkCmdPipelineBarrier(
            vkCommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            static_cast<uint32_t>(imageBarriers.size()),
            imageBarriers.data());
    }

    if (pipelineDesc.hasConstantData &&
        dispatchDesc.constantBufferData != nullptr &&
        dispatchDesc.constantBufferDataSize > 0 &&
        !dispatchDesc.constantBufferDataMatchesPreviousDispatch) {
        const NativeBuffer constantBuffer = nativeBuffer(*impl_->constantBuffer);
        if (constantBuffer.buffer == VK_NULL_HANDLE) {
            return makeError(Error::InvalidArgument);
        }
        VkBufferMemoryBarrier toTransfer{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = constantBuffer.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vkCmdPipelineBarrier(
            vkCommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            1,
            &toTransfer,
            0,
            nullptr);
        vkCmdUpdateBuffer(
            vkCommandBuffer,
            constantBuffer.buffer,
            0,
            dispatchDesc.constantBufferDataSize,
            dispatchDesc.constantBufferData);
        VkBufferMemoryBarrier toShader{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = constantBuffer.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vkCmdPipelineBarrier(
            vkCommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            1,
            &toShader,
            0,
            nullptr);
    }

    vkCmdBindPipeline(vkCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(
        vkCommandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipelineLayout,
        instanceDesc->constantBufferAndSamplersSpaceIndex,
        1,
        &impl_->samplerDescriptorSet,
        0,
        nullptr);
    vkCmdPushDescriptorSetKHR(
        vkCommandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipelineLayout,
        instanceDesc->resourcesSpaceIndex,
        static_cast<uint32_t>(writes.size()),
        writes.data());
    vkCmdDispatch(vkCommandBuffer, dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);
    return {};
}

#endif

} // namespace metallic::render::vulkan
