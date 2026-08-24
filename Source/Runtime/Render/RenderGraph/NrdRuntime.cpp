#include "Runtime/Render/RenderGraph/NrdRuntime.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace metallic::render {
namespace {

#if METALLIC_HAS_NRD

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

SamplerDesc samplerFromNrd(nrd::Sampler sampler)
{
    const SamplerFilter filter = sampler == nrd::Sampler::LINEAR_CLAMP
        ? SamplerFilter::Linear
        : SamplerFilter::Nearest;
    return SamplerDesc{
        .minFilter = filter,
        .magFilter = filter,
        .mipFilter = SamplerFilter::Nearest,
        .addressU = SamplerAddressMode::ClampToEdge,
        .addressV = SamplerAddressMode::ClampToEdge,
        .addressW = SamplerAddressMode::ClampToEdge,
        .minLod = 0.0f,
        .maxLod = 1000.0f,
    };
}

struct NrdPushData {
    uint64_t constantBufferAddress = 0;
    uint32_t samplerIndex = 0;
    uint32_t sampledImageIndex = 0;
    uint32_t storageImageIndex = 0;
    uint32_t padding = 0;
};

static_assert(offsetof(NrdPushData, constantBufferAddress) == 0);
static_assert(offsetof(NrdPushData, samplerIndex) == 8);
static_assert(offsetof(NrdPushData, sampledImageIndex) == 12);
static_assert(offsetof(NrdPushData, storageImageIndex) == 16);

#endif

} // namespace

Format nrdNormalRoughnessFormat()
{
#if METALLIC_HAS_NRD
    const nrd::LibraryDesc* libraryDesc = nrd::GetLibraryDesc();
    if (libraryDesc == nullptr) {
        return Format::Unknown;
    }
    switch (libraryDesc->normalEncoding) {
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

struct NrdRuntime::Impl {
    struct TextureResource {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> view;
        ResourceState state = ResourceState::Undefined;
    };

    nrd::Instance* instance = nullptr;
    Device* device = nullptr;
    uint16_t width = 0;
    uint16_t height = 0;
    std::vector<TextureResource> permanentTextures;
    std::vector<TextureResource> transientTextures;
    NrdUserTexturePool userTexturePool{};
    std::unique_ptr<BindlessHeap> descriptorHeap;
    std::vector<BindlessHandle> samplerHandles;
    std::vector<BindlessHandle> sampledImageHandles;
    std::vector<BindlessHandle> storageImageHandles;
    std::vector<std::unique_ptr<ComputePipeline>> pipelines;
    uint32_t sampledImageCursor = 0;
    uint32_t storageImageCursor = 0;
    uint64_t previousConstantAddress = 0;
    bool internalTexturesInitialized = false;
};

NrdRuntime::NrdRuntime()
    : impl_(std::make_unique<Impl>())
{
}

NrdRuntime::~NrdRuntime()
{
    clear();
}

NrdRuntime::NrdRuntime(NrdRuntime&&) noexcept = default;
NrdRuntime& NrdRuntime::operator=(NrdRuntime&&) noexcept = default;

Result NrdRuntime::initialize(
    Device& device,
    uint16_t width,
    uint16_t height,
    const NrdUserTexturePool& userTexturePool,
    std::string& log)
{
    if (width == 0 || height == 0) {
        log = "NrdRuntime requires a non-zero image size";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().bindlessDescriptorHeap) {
        log = "NrdRuntime requires bindless descriptor heaps";
        return makeError(Error::Unsupported);
    }

    clear();
    impl_ = std::make_unique<Impl>();
    impl_->device = &device;
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
    const nrd::InstanceCreationDesc creationDesc{
        .denoisers = denoisers.data(),
        .denoisersNum = static_cast<uint32_t>(denoisers.size()),
    };
    const nrd::Result createResult = nrd::CreateInstance(creationDesc, impl_->instance);
    if (createResult != nrd::Result::SUCCESS) {
        log = nrdResultMessage("nrd::CreateInstance", createResult);
        clear();
        return resultFromNrd(createResult);
    }

    const nrd::InstanceDesc* instanceDesc = nrd::GetInstanceDesc(*impl_->instance);
    const nrd::LibraryDesc* libraryDesc = nrd::GetLibraryDesc();
    if (instanceDesc == nullptr || libraryDesc == nullptr) {
        log = "NrdRuntime failed to get NRD descriptors";
        clear();
        return makeError(Error::Failure);
    }

    auto createTextureResource = [&](const nrd::TextureDesc& nrdDesc, Impl::TextureResource& resource) {
        const Format format = formatFromNrd(nrdDesc.format);
        if (format == Format::Unknown || nrdDesc.downsampleFactor == 0) {
            log = "NrdRuntime received an unsupported internal texture descriptor";
            return makeError(Error::Unsupported);
        }
        const uint16_t textureWidth = divideRoundUp(width, nrdDesc.downsampleFactor);
        const uint16_t textureHeight = divideRoundUp(height, nrdDesc.downsampleFactor);
        Result result = device.createTexture(
            TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::Sampled |
                    TextureUsageBits::Storage |
                    TextureUsageBits::TransferDestination,
                .format = format,
                .width = textureWidth,
                .height = textureHeight,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
            },
            resource.texture);
        if (!result || resource.texture == nullptr) {
            log = "createTexture(NRD internal) returned ";
            log += resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createTextureView(
            *resource.texture,
            TextureViewDesc{
                .format = format,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            resource.view);
        if (!result || resource.view == nullptr) {
            log = "createTextureView(NRD internal) returned ";
            log += resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        return Result{};
    };

    impl_->permanentTextures.resize(instanceDesc->permanentPoolSize);
    for (uint32_t index = 0; index < instanceDesc->permanentPoolSize; ++index) {
        Result result = createTextureResource(
            instanceDesc->permanentPool[index],
            impl_->permanentTextures[index]);
        if (!result) {
            clear();
            return result;
        }
    }
    impl_->transientTextures.resize(instanceDesc->transientPoolSize);
    for (uint32_t index = 0; index < instanceDesc->transientPoolSize; ++index) {
        Result result = createTextureResource(
            instanceDesc->transientPool[index],
            impl_->transientTextures[index]);
        if (!result) {
            clear();
            return result;
        }
    }

    const uint32_t sampledImageCount = std::max(
        instanceDesc->descriptorPoolDesc.totalTexturesNum,
        instanceDesc->descriptorPoolDesc.perSetTexturesMaxNum);
    const uint32_t storageImageCount = std::max(
        instanceDesc->descriptorPoolDesc.totalStorageTexturesNum,
        instanceDesc->descriptorPoolDesc.perSetStorageTexturesMaxNum);
    Result result = device.createBindlessHeap(
        BindlessHeapDesc{
            .maxSamplers = instanceDesc->samplersNum,
            .maxSampledImages = sampledImageCount,
            .maxStorageImages = storageImageCount,
        },
        impl_->descriptorHeap);
    if (!result || impl_->descriptorHeap == nullptr) {
        log = "createBindlessHeap(NRD) returned ";
        log += resultToString(result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    impl_->samplerHandles.resize(instanceDesc->samplersNum);
    std::vector<BindlessSamplerWrite> samplerWrites(instanceDesc->samplersNum);
    for (uint32_t index = 0; index < instanceDesc->samplersNum; ++index) {
        result = impl_->descriptorHeap->allocateSampler(impl_->samplerHandles[index]);
        if (!result) {
            log = "allocateSampler(NRD) returned ";
            log += resultToString(result);
            clear();
            return result;
        }
        samplerWrites[index] = {
            .handle = impl_->samplerHandles[index],
            .sampler = samplerFromNrd(instanceDesc->samplers[index]),
        };
    }
    if (!samplerWrites.empty()) {
        result = impl_->descriptorHeap->writeSamplers(
            samplerWrites.data(),
            static_cast<uint32_t>(samplerWrites.size()));
        if (!result) {
            log = "writeSamplers(NRD) returned ";
            log += resultToString(result);
            clear();
            return result;
        }
    }

    impl_->sampledImageHandles.resize(sampledImageCount);
    for (BindlessHandle& handle : impl_->sampledImageHandles) {
        result = impl_->descriptorHeap->allocateSampledImage(handle);
        if (!result) {
            log = "allocateSampledImage(NRD) returned ";
            log += resultToString(result);
            clear();
            return result;
        }
    }
    impl_->storageImageHandles.resize(storageImageCount);
    for (BindlessHandle& handle : impl_->storageImageHandles) {
        result = impl_->descriptorHeap->allocateStorageImage(handle);
        if (!result) {
            log = "allocateStorageImage(NRD) returned ";
            log += resultToString(result);
            clear();
            return result;
        }
    }

    const uint32_t samplerBinding = libraryDesc->spirvBindingOffsets.samplerOffset +
        instanceDesc->samplersBaseRegisterIndex;
    const uint32_t constantBinding = libraryDesc->spirvBindingOffsets.constantBufferOffset +
        instanceDesc->constantBufferRegisterIndex;
    const uint32_t sampledBinding = libraryDesc->spirvBindingOffsets.textureOffset +
        instanceDesc->resourcesBaseRegisterIndex;
    const uint32_t storageBinding = libraryDesc->spirvBindingOffsets.storageTextureAndBufferOffset +
        instanceDesc->resourcesBaseRegisterIndex;

    impl_->pipelines.resize(instanceDesc->pipelinesNum);
    for (uint32_t pipelineIndex = 0; pipelineIndex < instanceDesc->pipelinesNum; ++pipelineIndex) {
        const nrd::PipelineDesc& pipelineDesc = instanceDesc->pipelines[pipelineIndex];
        if (pipelineDesc.computeShaderSPIRV.bytecode == nullptr || pipelineDesc.computeShaderSPIRV.size == 0) {
            log = "NrdRuntime requires NRD SPIR-V shader blobs";
            clear();
            return makeError(Error::Unsupported);
        }

        std::vector<ShaderBindingMappingDesc> mappings;
        mappings.reserve(2 + pipelineDesc.resourceRangesNum);
        if (instanceDesc->samplersNum > 0) {
            mappings.push_back(ShaderBindingMappingDesc{
                .descriptorSet = instanceDesc->constantBufferAndSamplersSpaceIndex,
                .firstBinding = samplerBinding,
                .bindingCount = instanceDesc->samplersNum,
                .type = ShaderBindingType::Sampler,
                .source = ShaderBindingSource::HeapIndexFromPushData,
                .pushDataOffset = static_cast<uint32_t>(offsetof(NrdPushData, samplerIndex)),
            });
        }
        if (pipelineDesc.hasConstantData) {
            mappings.push_back(ShaderBindingMappingDesc{
                .descriptorSet = instanceDesc->constantBufferAndSamplersSpaceIndex,
                .firstBinding = constantBinding,
                .bindingCount = 1,
                .type = ShaderBindingType::ConstantBuffer,
                .source = ShaderBindingSource::DeviceAddressFromPushData,
                .pushDataOffset = static_cast<uint32_t>(offsetof(NrdPushData, constantBufferAddress)),
            });
        }
        for (uint32_t rangeIndex = 0; rangeIndex < pipelineDesc.resourceRangesNum; ++rangeIndex) {
            const nrd::ResourceRangeDesc& range = pipelineDesc.resourceRanges[rangeIndex];
            const bool storage = range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
            mappings.push_back(ShaderBindingMappingDesc{
                .descriptorSet = instanceDesc->resourcesSpaceIndex,
                .firstBinding = storage ? storageBinding : sampledBinding,
                .bindingCount = range.descriptorsNum,
                .type = storage ? ShaderBindingType::StorageImage : ShaderBindingType::SampledImage,
                .source = ShaderBindingSource::HeapIndexFromPushData,
                .pushDataOffset = storage
                    ? static_cast<uint32_t>(offsetof(NrdPushData, storageImageIndex))
                    : static_cast<uint32_t>(offsetof(NrdPushData, sampledImageIndex)),
            });
        }

        std::unique_ptr<ShaderModule> shader;
        result = device.createShaderModule(
            ShaderModuleDesc{
                .code = static_cast<const uint32_t*>(pipelineDesc.computeShaderSPIRV.bytecode),
                .byteSize = pipelineDesc.computeShaderSPIRV.size,
                .debugName = pipelineDesc.shaderIdentifier,
            },
            shader);
        if (!result || shader == nullptr) {
            log = "createShaderModule(NRD) returned ";
            log += resultToString(result);
            clear();
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createComputePipeline(
            ComputePipelineDesc{
                .computeShader = shader.get(),
                .computeEntryPoint = instanceDesc->shaderEntryPoint,
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(NrdPushData),
                .bindingMappings = mappings.data(),
                .bindingMappingCount = static_cast<uint32_t>(mappings.size()),
            },
            impl_->pipelines[pipelineIndex]);
        if (!result || impl_->pipelines[pipelineIndex] == nullptr) {
            log = "createComputePipeline(NRD) returned ";
            log += resultToString(result);
            clear();
            return result ? makeError(Error::Failure) : result;
        }
    }
    return {};
}

void NrdRuntime::clear()
{
    if (impl_ == nullptr) {
        return;
    }
    impl_->pipelines.clear();
    impl_->descriptorHeap.reset();
    impl_->permanentTextures.clear();
    impl_->transientTextures.clear();
    if (impl_->instance != nullptr) {
        nrd::DestroyInstance(*impl_->instance);
        impl_->instance = nullptr;
    }
    impl_->device = nullptr;
    impl_->width = 0;
    impl_->height = 0;
}

bool NrdRuntime::valid() const
{
    return impl_ != nullptr &&
        impl_->instance != nullptr &&
        impl_->device != nullptr &&
        impl_->descriptorHeap != nullptr &&
        !impl_->pipelines.empty();
}

uint16_t NrdRuntime::width() const
{
    return impl_ != nullptr ? impl_->width : 0;
}

uint16_t NrdRuntime::height() const
{
    return impl_ != nullptr ? impl_->height : 0;
}

void NrdRuntime::setUserPoolTexture(nrd::ResourceType resource, Texture& texture, TextureView& view)
{
    if (impl_ == nullptr || static_cast<size_t>(resource) >= impl_->userTexturePool.size()) {
        return;
    }
    impl_->userTexturePool[static_cast<size_t>(resource)] = {
        .texture = &texture,
        .view = &view,
    };
}

Result NrdRuntime::setCommonSettings(const nrd::CommonSettings& settings)
{
    if (!valid()) {
        return makeError(Error::InvalidArgument);
    }
    impl_->sampledImageCursor = 0;
    impl_->storageImageCursor = 0;
    impl_->previousConstantAddress = 0;
    return resultFromNrd(nrd::SetCommonSettings(*impl_->instance, settings));
}

Result NrdRuntime::setReblurSettings(const nrd::ReblurSettings& settings)
{
    return setDenoiserSettings(
        static_cast<nrd::Identifier>(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR),
        &settings);
}

Result NrdRuntime::setRelaxSettings(const nrd::RelaxSettings& settings)
{
    return setDenoiserSettings(
        static_cast<nrd::Identifier>(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR),
        &settings);
}

Result NrdRuntime::setDenoiserSettings(nrd::Identifier identifier, const void* settings)
{
    if (!valid() || settings == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return resultFromNrd(nrd::SetDenoiserSettings(*impl_->instance, identifier, settings));
}

Result NrdRuntime::denoise(NrdDenoiserMode mode, CommandBuffer& commandBuffer, Streamer& streamer)
{
    if (mode == NrdDenoiserMode::Reference) {
        const std::array<nrd::Identifier, 2> identifiers = {
            static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE),
            static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE) + 1,
        };
        return denoiseIdentifiers(
            identifiers.data(),
            static_cast<uint32_t>(identifiers.size()),
            commandBuffer,
            streamer);
    }

    const nrd::Identifier identifier = mode == NrdDenoiserMode::Relax
        ? static_cast<nrd::Identifier>(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR)
        : static_cast<nrd::Identifier>(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR);
    return denoiseIdentifiers(&identifier, 1, commandBuffer, streamer);
}

Result NrdRuntime::denoiseIdentifiers(
    const nrd::Identifier* denoisers,
    uint32_t denoiserCount,
    CommandBuffer& commandBuffer,
    Streamer& streamer)
{
    if (!valid() || denoisers == nullptr || denoiserCount == 0) {
        return makeError(Error::InvalidArgument);
    }

    if (!impl_->internalTexturesInitialized) {
        std::vector<TextureBarrierDesc> barriers;
        barriers.reserve(impl_->permanentTextures.size() + impl_->transientTextures.size());
        auto append = [&barriers](Impl::TextureResource& resource) {
            barriers.push_back(TextureBarrierDesc{
                .texture = resource.texture.get(),
                .before = resource.state,
                .after = ResourceState::TransferDestination,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            });
            resource.state = ResourceState::TransferDestination;
        };
        for (Impl::TextureResource& resource : impl_->permanentTextures) {
            append(resource);
        }
        for (Impl::TextureResource& resource : impl_->transientTextures) {
            append(resource);
        }
        if (!barriers.empty()) {
            commandBuffer.barrier(BarrierDesc{
                .textures = barriers.data(),
                .textureCount = static_cast<uint32_t>(barriers.size()),
            });
            const ColorValue clear{0.0f, 0.0f, 0.0f, 0.0f};
            for (Impl::TextureResource& resource : impl_->permanentTextures) {
                commandBuffer.clearColorTexture(*resource.texture, resource.state, clear);
            }
            for (Impl::TextureResource& resource : impl_->transientTextures) {
                commandBuffer.clearColorTexture(*resource.texture, resource.state, clear);
            }
        }
        impl_->internalTexturesInitialized = true;
    }

    const nrd::DispatchDesc* dispatches = nullptr;
    uint32_t dispatchCount = 0;
    const nrd::Result nrdResult = nrd::GetComputeDispatches(
        *impl_->instance,
        denoisers,
        denoiserCount,
        dispatches,
        dispatchCount);
    if (nrdResult != nrd::Result::SUCCESS) {
        return resultFromNrd(nrdResult);
    }

    commandBuffer.bindBindlessHeap(*impl_->descriptorHeap);
    for (uint32_t dispatchIndex = 0; dispatchIndex < dispatchCount; ++dispatchIndex) {
        const nrd::DispatchDesc& dispatchDesc = dispatches[dispatchIndex];
        commandBuffer.beginDebugLabel(DebugLabelDesc{
            .name = dispatchDesc.name != nullptr ? dispatchDesc.name : "NRD",
            .color = ColorValue{0.2f, 0.8f, 0.25f, 1.0f},
        });
        Result result = dispatch(commandBuffer, streamer, dispatchDesc, impl_->previousConstantAddress);
        commandBuffer.endDebugLabel();
        if (!result) {
            return result;
        }
    }
    return {};
}

Result NrdRuntime::dispatch(
    CommandBuffer& commandBuffer,
    Streamer& streamer,
    const nrd::DispatchDesc& dispatchDesc,
    uint64_t& previousConstantAddress)
{
    if (!valid() || dispatchDesc.pipelineIndex >= impl_->pipelines.size()) {
        return makeError(Error::InvalidArgument);
    }
    const nrd::InstanceDesc* instanceDesc = nrd::GetInstanceDesc(*impl_->instance);
    if (instanceDesc == nullptr) {
        return makeError(Error::Failure);
    }
    const nrd::PipelineDesc& pipelineDesc = instanceDesc->pipelines[dispatchDesc.pipelineIndex];

    struct ResolvedTexture {
        Texture* texture = nullptr;
        TextureView* view = nullptr;
        ResourceState* state = nullptr;
    };
    auto resolve = [&](const nrd::ResourceDesc& resourceDesc) -> ResolvedTexture {
        if (resourceDesc.type == nrd::ResourceType::TRANSIENT_POOL) {
            if (resourceDesc.indexInPool >= impl_->transientTextures.size()) {
                return {};
            }
            Impl::TextureResource& resource = impl_->transientTextures[resourceDesc.indexInPool];
            return {resource.texture.get(), resource.view.get(), &resource.state};
        }
        if (resourceDesc.type == nrd::ResourceType::PERMANENT_POOL) {
            if (resourceDesc.indexInPool >= impl_->permanentTextures.size()) {
                return {};
            }
            Impl::TextureResource& resource = impl_->permanentTextures[resourceDesc.indexInPool];
            return {resource.texture.get(), resource.view.get(), &resource.state};
        }
        const size_t index = static_cast<size_t>(resourceDesc.type);
        if (index >= impl_->userTexturePool.size()) {
            return {};
        }
        const NrdTextureRef& resource = impl_->userTexturePool[index];
        return {resource.texture, resource.view, nullptr};
    };

    uint32_t sampledCount = 0;
    uint32_t storageCount = 0;
    for (uint32_t rangeIndex = 0; rangeIndex < pipelineDesc.resourceRangesNum; ++rangeIndex) {
        const nrd::ResourceRangeDesc& range = pipelineDesc.resourceRanges[rangeIndex];
        if (range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE) {
            storageCount += range.descriptorsNum;
        } else {
            sampledCount += range.descriptorsNum;
        }
    }
    if (impl_->sampledImageCursor > impl_->sampledImageHandles.size() ||
        sampledCount > impl_->sampledImageHandles.size() - impl_->sampledImageCursor ||
        impl_->storageImageCursor > impl_->storageImageHandles.size() ||
        storageCount > impl_->storageImageHandles.size() - impl_->storageImageCursor) {
        return makeError(Error::OutOfMemory);
    }

    const uint32_t sampledBase = impl_->sampledImageCursor;
    const uint32_t storageBase = impl_->storageImageCursor;
    std::vector<BindlessImageWrite> writes;
    writes.reserve(dispatchDesc.resourcesNum);
    std::vector<TextureBarrierDesc> barriers;
    barriers.reserve(dispatchDesc.resourcesNum);
    std::unordered_set<Texture*> transitioned;

    uint32_t resourceIndex = 0;
    for (uint32_t rangeIndex = 0; rangeIndex < pipelineDesc.resourceRangesNum; ++rangeIndex) {
        const nrd::ResourceRangeDesc& range = pipelineDesc.resourceRanges[rangeIndex];
        const bool storage = range.descriptorType == nrd::DescriptorType::STORAGE_TEXTURE;
        for (uint32_t descriptorIndex = 0; descriptorIndex < range.descriptorsNum; ++descriptorIndex) {
            if (resourceIndex >= dispatchDesc.resourcesNum) {
                return makeError(Error::InvalidArgument);
            }
            const nrd::ResourceDesc& resourceDesc = dispatchDesc.resources[resourceIndex++];
            if (resourceDesc.descriptorType != range.descriptorType) {
                return makeError(Error::InvalidArgument);
            }
            ResolvedTexture resource = resolve(resourceDesc);
            if (resource.texture == nullptr || resource.view == nullptr) {
                return makeError(Error::InvalidArgument);
            }

            const BindlessHandle handle = storage
                ? impl_->storageImageHandles[impl_->storageImageCursor++]
                : impl_->sampledImageHandles[impl_->sampledImageCursor++];
            writes.push_back(BindlessImageWrite{
                .handle = handle,
                .view = resource.view,
                .state = ResourceState::General,
            });
            if (transitioned.insert(resource.texture).second) {
                barriers.push_back(TextureBarrierDesc{
                    .texture = resource.texture,
                    .before = resource.state != nullptr ? *resource.state : ResourceState::General,
                    .after = ResourceState::General,
                    .baseMip = 0,
                    .mipCount = resource.texture->desc().mipCount,
                    .baseLayer = 0,
                    .layerCount = resource.texture->desc().layerCount,
                });
            }
            if (resource.state != nullptr) {
                *resource.state = ResourceState::General;
            }
        }
    }
    if (resourceIndex != dispatchDesc.resourcesNum) {
        return makeError(Error::InvalidArgument);
    }
    if (!barriers.empty()) {
        commandBuffer.barrier(BarrierDesc{
            .textures = barriers.data(),
            .textureCount = static_cast<uint32_t>(barriers.size()),
        });
    }
    if (!writes.empty()) {
        Result result = impl_->descriptorHeap->writeImages(
            writes.data(),
            static_cast<uint32_t>(writes.size()));
        if (!result) {
            return result;
        }
    }

    if (pipelineDesc.hasConstantData) {
        if (!dispatchDesc.constantBufferDataMatchesPreviousDispatch || previousConstantAddress == 0) {
            if (dispatchDesc.constantBufferData == nullptr || dispatchDesc.constantBufferDataSize == 0) {
                return makeError(Error::InvalidArgument);
            }
            const uint64_t offset = streamer.streamConstantData(
                dispatchDesc.constantBufferData,
                dispatchDesc.constantBufferDataSize);
            Buffer* constantBuffer = streamer.constantBuffer();
            if (offset == std::numeric_limits<uint64_t>::max() || constantBuffer == nullptr) {
                return makeError(Error::OutOfMemory);
            }
            const uint64_t baseAddress = constantBuffer->deviceAddress();
            if (baseAddress == 0 || offset > std::numeric_limits<uint64_t>::max() - baseAddress) {
                return makeError(Error::Failure);
            }
            previousConstantAddress = baseAddress + offset;
        }
    }

    const NrdPushData push{
        .constantBufferAddress = previousConstantAddress,
        .samplerIndex = impl_->samplerHandles.empty() ? 0 : impl_->samplerHandles.front().index,
        .sampledImageIndex = sampledCount == 0
            ? 0
            : impl_->sampledImageHandles[sampledBase].index,
        .storageImageIndex = storageCount == 0
            ? 0
            : impl_->storageImageHandles[storageBase].index,
    };
    ComputePipeline* pipeline = impl_->pipelines[dispatchDesc.pipelineIndex].get();
    if (pipeline == nullptr) {
        return makeError(Error::Failure);
    }
    commandBuffer.bindComputePipeline(*pipeline, &push, sizeof(push));
    commandBuffer.dispatch(dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);
    return {};
}

#endif

} // namespace metallic::render
