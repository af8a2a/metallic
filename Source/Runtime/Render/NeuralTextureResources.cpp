#include "Runtime/Render/NeuralTextureResources.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef METALLIC_HAS_NTC
#define METALLIC_HAS_NTC 0
#endif

#if METALLIC_HAS_NTC
#include <libntc/ntc.h>
#include <libntc/wrappers.h>
#endif

namespace metallic::render {
namespace {

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    alignment = std::max<uint64_t>(alignment, 1);
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

void appendWarning(std::string& log, std::string message)
{
    spdlog::warn("{}", message);
    if (!log.empty() && log.back() != '\n') {
        log += '\n';
    }
    log += std::move(message);
    log += '\n';
}

Result createUploadBuffer(
    Device& device,
    const void* data,
    uint64_t byteSize,
    std::unique_ptr<Buffer>& outBuffer)
{
    if (data == nullptr || byteSize == 0) {
        return makeError(Error::InvalidArgument);
    }
    Result result = device.createBuffer(
        BufferDesc{
            .size = byteSize,
            .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy,
        },
        outBuffer);
    if (!result || outBuffer == nullptr) {
        return result ? makeError(Error::Failure) : result;
    }
    void* mapped = outBuffer->map();
    if (mapped == nullptr) {
        outBuffer.reset();
        return makeError(Error::Failure);
    }
    std::memcpy(mapped, data, static_cast<size_t>(byteSize));
    outBuffer->flush(0, byteSize);
    outBuffer->unmap();
    return {};
}

Result createDeviceStorageBuffer(
    Device& device,
    uint64_t byteSize,
    uint32_t structureStride,
    std::unique_ptr<Buffer>& outBuffer)
{
    return device.createBuffer(
        BufferDesc{
            .size = byteSize,
            .structureStride = structureStride,
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy,
        },
        outBuffer);
}

uint64_t rgba8MipChainByteSize(uint32_t width, uint32_t height, uint32_t mipCount)
{
    uint64_t byteSize = 0;
    for (uint32_t mip = 0; mip < mipCount; ++mip) {
        byteSize += static_cast<uint64_t>(width) * height * 4u;
        width = std::max(width >> 1u, 1u);
        height = std::max(height >> 1u, 1u);
    }
    return byteSize;
}

} // namespace

struct NeuralTextureResources::Impl {
    struct TextureUpload {
        uint64_t bufferOffset = 0;
        uint64_t byteSize = 0;
        uint32_t rowPitch = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t mipLevel = 0;
        uint32_t layer = 0;
    };

    struct TextureSet {
        std::unique_ptr<Buffer> uploadBuffer;
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> view;
        std::vector<TextureUpload> uploads;
        ResourceState state = ResourceState::Undefined;
        uint64_t latentBytes = 0;
    };

    void clear()
    {
        sets.clear();
        latentViews.fill(nullptr);
        constantsBuffer.reset();
        constantsUpload.reset();
        weightsBuffer.reset();
        weightsUpload.reset();
        setInfoBuffer.reset();
        setInfoUpload.reset();
        logicalTextureSetMap.clear();
        stats = {};
        uploadRecorded = false;
    }

#if METALLIC_HAS_NTC
    struct PendingSet {
        int32_t imageIndex = scene::kInvalidSceneIndex;
        NtcTextureSetConstants constants{};
        std::vector<uint8_t> weights;
        std::vector<uint8_t> latentData;
        std::vector<TextureUpload> uploads;
        ntc::LatentTextureDesc desc{};
        uint64_t conventionalTextureBytes = 0;
    };

    bool readSet(
        ntc::IContext& context,
        const scene::Scene& loadedScene,
        int32_t imageIndex,
        PendingSet& outSet,
        std::string& error)
    {
        if (imageIndex < 0 || static_cast<size_t>(imageIndex) >= loadedScene.images().size()) {
            error = "NTC image index is out of range";
            return false;
        }
        const scene::RenderImage& image = loadedScene.images()[static_cast<size_t>(imageIndex)];
        ntc::FileStreamWrapper fileStream(&context);
        ntc::MemoryStreamWrapper memoryStream(&context);
        ntc::Status status = ntc::Status::InvalidArgument;
        if (!image.encodedData.empty()) {
            status = context.OpenReadOnlyMemory(
                image.encodedData.data(),
                image.encodedData.size(),
                memoryStream.ptr());
        } else if (!image.uri.empty() && image.uri.rfind("data:", 0) != 0) {
            std::filesystem::path path = image.uri;
            if (path.is_relative()) {
                path = loadedScene.filename().parent_path() / path;
            }
            const std::string pathString = path.lexically_normal().string();
            status = context.OpenFile(pathString.c_str(), false, fileStream.ptr());
        }
        ntc::IStream* stream = fileStream.Get() != nullptr
            ? fileStream.Get()
            : memoryStream.Get();
        if (status != ntc::Status::Ok || stream == nullptr) {
            error = "failed to open NTC image '" + image.uri + "': " +
                ntc::StatusToString(status) + " (" + ntc::GetLastErrorMessage() + ")";
            return false;
        }

        ntc::TextureSetMetadataWrapper metadata(&context);
        status = context.CreateTextureSetMetadataFromStream(stream, metadata.ptr());
        if (status != ntc::Status::Ok || metadata.Get() == nullptr) {
            error = "failed to parse NTC image '" + image.uri + "': " +
                ntc::StatusToString(status) + " (" + ntc::GetLastErrorMessage() + ")";
            return false;
        }

        ntc::InferenceData inferenceData;
        status = context.MakeInferenceData(
            metadata.Get(),
            ntc::InferenceWeightType::GenericInt8,
            0,
            &inferenceData);
        if (status != ntc::Status::Ok) {
            error = "failed to create NTC inference data for '" + image.uri + "': " +
                ntc::StatusToString(status) + " (" + ntc::GetLastErrorMessage() + ")";
            return false;
        }
        outSet.constants = inferenceData.constants;
        const ntc::TextureSetDesc colorDesc = metadata->GetDesc();
        if (colorDesc.width <= 0 || colorDesc.height <= 0 || colorDesc.mips <= 0) {
            error = "NTC image has an invalid color texture description: " + image.uri;
            return false;
        }
        outSet.conventionalTextureBytes = rgba8MipChainByteSize(
            static_cast<uint32_t>(colorDesc.width),
            static_cast<uint32_t>(colorDesc.height),
            static_cast<uint32_t>(colorDesc.mips));

        const void* weights = nullptr;
        size_t weightSize = 0;
        size_t convertedWeightSize = 0;
        status = metadata->GetInferenceWeights(
            ntc::InferenceWeightType::GenericInt8,
            &weights,
            &weightSize,
            &convertedWeightSize);
        if (status != ntc::Status::Ok || weights == nullptr || weightSize == 0 ||
            convertedWeightSize != 0) {
            error = "failed to read Generic INT8 NTC weights for '" + image.uri + "': " +
                ntc::StatusToString(status) + " (" + ntc::GetLastErrorMessage() + ")";
            return false;
        }
        outSet.weights.assign(
            static_cast<const uint8_t*>(weights),
            static_cast<const uint8_t*>(weights) + weightSize);
        outSet.desc = metadata->GetLatentTextureDesc();
        if (outSet.desc.width <= 0 || outSet.desc.height <= 0 ||
            outSet.desc.arraySize <= 0 || outSet.desc.mipLevels <= 0) {
            error = "NTC image has an invalid latent texture description: " + image.uri;
            return false;
        }

        const uint64_t alignment = 4;
        for (int mipLevel = 0; mipLevel < outSet.desc.mipLevels; ++mipLevel) {
            for (int layer = 0; layer < outSet.desc.arraySize; ++layer) {
                ntc::LatentTextureFootprint footprint;
                status = metadata->GetLatentTextureFootprint(mipLevel, layer, footprint);
                if (status != ntc::Status::Ok || footprint.width <= 0 ||
                    footprint.height <= 0 || footprint.rowPitch == 0 ||
                    footprint.rowPitch > std::numeric_limits<uint32_t>::max() ||
                    footprint.buffer.rangeInStream.size > std::numeric_limits<size_t>::max() ||
                    footprint.buffer.uncompressedSize > std::numeric_limits<size_t>::max()) {
                    error = "NTC latent footprint is invalid for '" + image.uri + "'";
                    return false;
                }
                std::vector<uint8_t> encoded(
                    static_cast<size_t>(footprint.buffer.rangeInStream.size));
                if (!stream->Seek(footprint.buffer.rangeInStream.offset) ||
                    !stream->Read(encoded.data(), encoded.size())) {
                    error = "failed to read NTC latent data for '" + image.uri + "'";
                    return false;
                }
                std::vector<uint8_t> decoded;
                const uint8_t* data = encoded.data();
                size_t dataSize = encoded.size();
                if (footprint.buffer.compressionType != ntc::CompressionType::None) {
                    decoded.resize(static_cast<size_t>(footprint.buffer.uncompressedSize));
                    status = context.DecompressBuffer(
                        footprint.buffer.compressionType,
                        encoded.data(),
                        encoded.size(),
                        decoded.data(),
                        decoded.size(),
                        footprint.buffer.uncompressedCrc32);
                    if (status != ntc::Status::Ok) {
                        error = "failed to decompress NTC latent data for '" + image.uri + "': " +
                            ntc::StatusToString(status) + " (" + ntc::GetLastErrorMessage() + ")";
                        return false;
                    }
                    data = decoded.data();
                    dataSize = decoded.size();
                }
                const uint64_t requiredSize =
                    static_cast<uint64_t>(footprint.rowPitch) * footprint.height;
                if (requiredSize > dataSize) {
                    error = "NTC latent payload is smaller than its footprint for '" + image.uri + "'";
                    return false;
                }
                const uint64_t offset = alignUp(outSet.latentData.size(), alignment);
                if (offset > std::numeric_limits<size_t>::max() - requiredSize) {
                    error = "NTC latent upload size overflow for '" + image.uri + "'";
                    return false;
                }
                outSet.latentData.resize(static_cast<size_t>(offset + requiredSize));
                std::memcpy(outSet.latentData.data() + offset, data, static_cast<size_t>(requiredSize));
                outSet.uploads.push_back(TextureUpload{
                    .bufferOffset = offset,
                    .byteSize = requiredSize,
                    .rowPitch = static_cast<uint32_t>(footprint.rowPitch),
                    .width = static_cast<uint32_t>(footprint.width),
                    .height = static_cast<uint32_t>(footprint.height),
                    .mipLevel = static_cast<uint32_t>(mipLevel),
                    .layer = static_cast<uint32_t>(layer),
                });
            }
        }
        outSet.imageIndex = imageIndex;
        return true;
    }
#endif

    std::vector<TextureSet> sets;
    std::array<TextureView*, kMaxNeuralTextureSets> latentViews{};
    std::unique_ptr<Buffer> constantsBuffer;
    std::unique_ptr<Buffer> constantsUpload;
    std::unique_ptr<Buffer> weightsBuffer;
    std::unique_ptr<Buffer> weightsUpload;
    std::unique_ptr<Buffer> setInfoBuffer;
    std::unique_ptr<Buffer> setInfoUpload;
    std::vector<uint32_t> logicalTextureSetMap;
    SamplerDesc sampler{
        .minFilter = SamplerFilter::Linear,
        .magFilter = SamplerFilter::Linear,
        .mipFilter = SamplerFilter::Linear,
        .addressU = SamplerAddressMode::Repeat,
        .addressV = SamplerAddressMode::Repeat,
        .addressW = SamplerAddressMode::Repeat,
    };
    NeuralTextureMemoryStats stats;
    bool uploadRecorded = false;
};

NeuralTextureResources::NeuralTextureResources()
    : impl_(std::make_unique<Impl>())
{
}

NeuralTextureResources::~NeuralTextureResources() = default;
NeuralTextureResources::NeuralTextureResources(NeuralTextureResources&&) noexcept = default;
NeuralTextureResources& NeuralTextureResources::operator=(NeuralTextureResources&&) noexcept = default;

Result NeuralTextureResources::prepare(
    Device& device,
    const scene::Scene& loadedScene,
    std::string& log)
{
    impl_->clear();
    impl_->logicalTextureSetMap.assign(
        loadedScene.textures().size(),
        kInvalidNeuralTextureSetIndex);

#if !METALLIC_HAS_NTC
    (void)device;
    (void)loadedScene;
    (void)log;
    return {};
#else
    std::unordered_set<int32_t> referencedImages;
    for (const scene::RenderTexture& texture : loadedScene.textures()) {
        if (texture.hasNeuralSource()) {
            referencedImages.insert(texture.ntcImageIndex);
        }
    }
    if (referencedImages.empty()) {
        return {};
    }
    if (!device.capabilities().shaderIntegerDotProduct) {
        appendWarning(
            log,
            "[NTC] Generic INT8 inference requires Vulkan shaderIntegerDotProduct; using conventional textures");
        return {};
    }

    ntc::ContextWrapper context;
    ntc::ContextParameters contextParameters;
    contextParameters.cudaDevice = ntc::DisableCudaDevice;
    contextParameters.enableCooperativeVector = false;
    const ntc::Status contextStatus = ntc::CreateContext(context.ptr(), contextParameters);
    if ((contextStatus != ntc::Status::Ok &&
         contextStatus != ntc::Status::CudaUnavailable) ||
        context.Get() == nullptr) {
        appendWarning(
            log,
            std::string("[NTC] LibNTC context unavailable; using conventional textures: ") +
                ntc::StatusToString(contextStatus) + " (" + ntc::GetLastErrorMessage() + ")");
        return {};
    }

    std::vector<Impl::PendingSet> pendingSets;
    std::unordered_map<int32_t, uint32_t> imageToSet;
    for (int32_t imageIndex : referencedImages) {
        if (pendingSets.size() >= kMaxNeuralTextureSets) {
            appendWarning(log, "[NTC] Texture-set limit reached; remaining sets use conventional textures");
            break;
        }
        Impl::PendingSet pending;
        std::string error;
        if (!impl_->readSet(*context.Get(), loadedScene, imageIndex, pending, error)) {
            appendWarning(log, "[NTC] " + error + "; using conventional textures");
            continue;
        }
        imageToSet[imageIndex] = static_cast<uint32_t>(pendingSets.size());
        pendingSets.push_back(std::move(pending));
    }
    if (pendingSets.empty()) {
        return {};
    }

    std::vector<NtcTextureSetConstants> constants;
    std::vector<uint8_t> weights;
    struct SetInfo {
        uint32_t weightOffset = 0;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
        uint32_t padding2 = 0;
    };
    static_assert(sizeof(SetInfo) == 16);
    std::vector<SetInfo> setInfos;
    constants.reserve(pendingSets.size());
    setInfos.reserve(pendingSets.size());
    impl_->sets.reserve(pendingSets.size());

    for (Impl::PendingSet& pending : pendingSets) {
        const uint64_t weightOffset = alignUp(weights.size(), 4);
        if (weightOffset > UINT32_MAX ||
            weightOffset > std::numeric_limits<size_t>::max() - pending.weights.size()) {
            appendWarning(log, "[NTC] Weight buffer exceeded the 32-bit shader offset range; using conventional textures");
            impl_->clear();
            return {};
        }
        weights.resize(static_cast<size_t>(weightOffset));
        weights.insert(weights.end(), pending.weights.begin(), pending.weights.end());
        constants.push_back(pending.constants);
        setInfos.push_back(SetInfo{.weightOffset = static_cast<uint32_t>(weightOffset)});

        Impl::TextureSet set;
        set.uploads = std::move(pending.uploads);
        set.latentBytes = pending.latentData.size();
        Result result = createUploadBuffer(
            device,
            pending.latentData.data(),
            pending.latentData.size(),
            set.uploadBuffer);
        if (!result) {
            appendWarning(log, "[NTC] Failed to allocate latent upload buffer; using conventional textures");
            impl_->clear();
            return {};
        }
        // The inference shader expects the exact packed BGRA4 latent representation.
        result = device.createTexture(
            TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::Bgra4Unorm,
                .width = static_cast<uint32_t>(pending.desc.width),
                .height = static_cast<uint32_t>(pending.desc.height),
                .depth = 1,
                .mipCount = static_cast<uint32_t>(pending.desc.mipLevels),
                .layerCount = static_cast<uint32_t>(pending.desc.arraySize),
                .memoryLocation = MemoryLocation::Device,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy,
            },
            set.texture);
        if (!result || set.texture == nullptr) {
            appendWarning(log, "[NTC] Failed to create BGRA4 latent texture; using conventional textures");
            impl_->clear();
            return {};
        }
        result = device.createTextureView(
            *set.texture,
            TextureViewDesc{
                .format = Format::Bgra4Unorm,
                .baseMip = 0,
                .mipCount = static_cast<uint32_t>(pending.desc.mipLevels),
                .baseLayer = 0,
                .layerCount = static_cast<uint32_t>(pending.desc.arraySize),
            },
            set.view);
        if (!result || set.view == nullptr) {
            appendWarning(log, "[NTC] Failed to create latent texture view; using conventional textures");
            impl_->clear();
            return {};
        }
        impl_->sets.push_back(std::move(set));
    }

    const auto createBufferPair = [&](const void* data,
                                      uint64_t byteSize,
                                      uint32_t stride,
                                      std::unique_ptr<Buffer>& upload,
                                      std::unique_ptr<Buffer>& destination) -> Result {
        Result result = createUploadBuffer(device, data, byteSize, upload);
        if (!result) {
            return result;
        }
        return createDeviceStorageBuffer(device, byteSize, stride, destination);
    };
    Result result = createBufferPair(
        constants.data(),
        constants.size() * sizeof(NtcTextureSetConstants),
        sizeof(NtcTextureSetConstants),
        impl_->constantsUpload,
        impl_->constantsBuffer);
    if (result) {
        result = createBufferPair(
            weights.data(),
            weights.size(),
            0,
            impl_->weightsUpload,
            impl_->weightsBuffer);
    }
    if (result) {
        result = createBufferPair(
            setInfos.data(),
            setInfos.size() * sizeof(SetInfo),
            sizeof(SetInfo),
            impl_->setInfoUpload,
            impl_->setInfoBuffer);
    }
    if (!result || impl_->constantsBuffer == nullptr || impl_->weightsBuffer == nullptr ||
        impl_->setInfoBuffer == nullptr) {
        appendWarning(log, "[NTC] Failed to allocate inference buffers; using conventional textures");
        impl_->clear();
        return {};
    }

    for (uint32_t textureIndex = 0; textureIndex < loadedScene.textures().size(); ++textureIndex) {
        const scene::RenderTexture& texture = loadedScene.textures()[textureIndex];
        const auto found = imageToSet.find(texture.ntcImageIndex);
        if (texture.hasNeuralSource() && found != imageToSet.end()) {
            impl_->logicalTextureSetMap[textureIndex] = found->second;
            ++impl_->stats.replacedTextureCount;
        }
    }
    std::unordered_map<int32_t, uint32_t> replacedConventionalImages;
    for (uint32_t textureIndex = 0; textureIndex < loadedScene.textures().size(); ++textureIndex) {
        const uint32_t setIndex = impl_->logicalTextureSetMap[textureIndex];
        if (setIndex != kInvalidNeuralTextureSetIndex) {
            replacedConventionalImages.try_emplace(
                loadedScene.textures()[textureIndex].imageIndex,
                setIndex);
        }
    }
    for (const auto& [imageIndex, setIndex] : replacedConventionalImages) {
        if (imageIndex >= 0 && static_cast<size_t>(imageIndex) < loadedScene.images().size() &&
            setIndex < pendingSets.size()) {
            impl_->stats.conventionalTextureBytes +=
                pendingSets[setIndex].conventionalTextureBytes;
        }
    }
    for (const Impl::TextureSet& set : impl_->sets) {
        impl_->stats.latentTextureBytes += set.latentBytes;
    }
    impl_->stats.textureSetCount = static_cast<uint32_t>(impl_->sets.size());
    impl_->stats.weightBytes = weights.size();
    impl_->stats.metadataBytes =
        constants.size() * sizeof(NtcTextureSetConstants) + setInfos.size() * sizeof(SetInfo);
    impl_->latentViews.fill(impl_->sets.front().view.get());
    for (uint32_t setIndex = 0; setIndex < impl_->sets.size(); ++setIndex) {
        impl_->latentViews[setIndex] = impl_->sets[setIndex].view.get();
    }

    const int64_t savedBytes = static_cast<int64_t>(impl_->stats.conventionalTextureBytes) -
        static_cast<int64_t>(impl_->stats.neuralBytes());
    const double reductionPercent = impl_->stats.conventionalTextureBytes > 0
        ? 100.0 * static_cast<double>(savedBytes) /
            static_cast<double>(impl_->stats.conventionalTextureBytes)
        : 0.0;
    spdlog::info(
        "[NTC] Prepared sets={} replacedTextures={} conventionalBytes={} neuralBytes={} savedBytes={} reduction={:.1f}%",
        impl_->stats.textureSetCount,
        impl_->stats.replacedTextureCount,
        impl_->stats.conventionalTextureBytes,
        impl_->stats.neuralBytes(),
        savedBytes,
        reductionPercent);
    return {};
#endif
}

Result NeuralTextureResources::recordUploads(CommandBuffer& commandBuffer)
{
    if (!active() || impl_->uploadRecorded) {
        return {};
    }
    for (Impl::TextureSet& set : impl_->sets) {
        TextureBarrierDesc toTransfer{
            .texture = set.texture.get(),
            .before = set.state,
            .after = ResourceState::TransferDestination,
            .baseMip = 0,
            .mipCount = set.texture->desc().mipCount,
            .baseLayer = 0,
            .layerCount = set.texture->desc().layerCount,
        };
        commandBuffer.barrier(BarrierDesc{.textures = &toTransfer, .textureCount = 1});
        set.state = ResourceState::TransferDestination;
        for (const Impl::TextureUpload& upload : set.uploads) {
            commandBuffer.copyBufferToTexture(BufferTextureCopyDesc{
                .buffer = set.uploadBuffer.get(),
                .texture = set.texture.get(),
                .bufferOffset = upload.bufferOffset,
                .bufferRowPitch = upload.rowPitch,
                .bufferSlicePitch = upload.rowPitch * upload.height,
                .width = upload.width,
                .height = upload.height,
                .depth = 1,
                .mipLevel = upload.mipLevel,
                .baseLayer = upload.layer,
                .layerCount = 1,
            });
        }
        TextureBarrierDesc toShaderRead{
            .texture = set.texture.get(),
            .before = ResourceState::TransferDestination,
            .after = ResourceState::ShaderRead,
            .baseMip = 0,
            .mipCount = set.texture->desc().mipCount,
            .baseLayer = 0,
            .layerCount = set.texture->desc().layerCount,
        };
        commandBuffer.barrier(BarrierDesc{.textures = &toShaderRead, .textureCount = 1});
        set.state = ResourceState::ShaderRead;
    }

    const struct BufferPair {
        Buffer* upload;
        Buffer* destination;
    } bufferPairs[] = {
        {impl_->constantsUpload.get(), impl_->constantsBuffer.get()},
        {impl_->weightsUpload.get(), impl_->weightsBuffer.get()},
        {impl_->setInfoUpload.get(), impl_->setInfoBuffer.get()},
    };
    for (const BufferPair& pair : bufferPairs) {
        if (pair.upload == nullptr || pair.destination == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        commandBuffer.copyBuffer(BufferCopyDesc{
            .source = pair.upload,
            .destination = pair.destination,
            .size = pair.destination->desc().size,
        });
        BufferBarrierDesc toGeneral{
            .buffer = pair.destination,
            .before = ResourceState::TransferDestination,
            .after = ResourceState::General,
            .offset = 0,
            .size = pair.destination->desc().size,
        };
        commandBuffer.barrier(BarrierDesc{.buffers = &toGeneral, .bufferCount = 1});
    }
    impl_->uploadRecorded = true;
    return {};
}

void NeuralTextureResources::releaseUploadBuffers()
{
    if (!impl_->uploadRecorded) {
        return;
    }
    for (Impl::TextureSet& set : impl_->sets) {
        set.uploadBuffer.reset();
    }
    impl_->constantsUpload.reset();
    impl_->weightsUpload.reset();
    impl_->setInfoUpload.reset();
}

void NeuralTextureResources::clear()
{
    impl_->clear();
}

bool NeuralTextureResources::active() const
{
    return !impl_->sets.empty() && impl_->constantsBuffer != nullptr &&
        impl_->weightsBuffer != nullptr && impl_->setInfoBuffer != nullptr;
}

bool NeuralTextureResources::uploaded() const
{
    return !active() || impl_->uploadRecorded;
}

uint64_t NeuralTextureResources::pendingUploadByteSize() const
{
    if (!active() || impl_->uploadRecorded) {
        return 0;
    }
    uint64_t bytes = 0;
    for (const Impl::TextureSet& set : impl_->sets) {
        bytes += set.latentBytes;
    }
    bytes += impl_->constantsBuffer->desc().size;
    bytes += impl_->weightsBuffer->desc().size;
    bytes += impl_->setInfoBuffer->desc().size;
    return bytes;
}

uint32_t NeuralTextureResources::pendingUploadRegionCount() const
{
    if (!active() || impl_->uploadRecorded) {
        return 0;
    }
    uint64_t regions = 3;
    for (const Impl::TextureSet& set : impl_->sets) {
        regions += set.uploads.size();
    }
    return static_cast<uint32_t>(std::min<uint64_t>(regions, UINT32_MAX));
}

uint32_t NeuralTextureResources::textureSetCount() const
{
    return static_cast<uint32_t>(impl_->sets.size());
}

uint32_t NeuralTextureResources::logicalTextureSetIndex(uint32_t logicalTextureIndex) const
{
    return logicalTextureIndex < impl_->logicalTextureSetMap.size()
        ? impl_->logicalTextureSetMap[logicalTextureIndex]
        : kInvalidNeuralTextureSetIndex;
}

const std::vector<uint32_t>& NeuralTextureResources::logicalTextureSetIndices() const
{
    return impl_->logicalTextureSetMap;
}

const std::array<TextureView*, kMaxNeuralTextureSets>&
NeuralTextureResources::latentTextureViews() const
{
    return impl_->latentViews;
}

Buffer* NeuralTextureResources::constantsBuffer() const
{
    return impl_->constantsBuffer.get();
}

Buffer* NeuralTextureResources::weightsBuffer() const
{
    return impl_->weightsBuffer.get();
}

Buffer* NeuralTextureResources::setInfoBuffer() const
{
    return impl_->setInfoBuffer.get();
}

const SamplerDesc& NeuralTextureResources::latentSampler() const
{
    return impl_->sampler;
}

const NeuralTextureMemoryStats& NeuralTextureResources::memoryStats() const
{
    return impl_->stats;
}

} // namespace metallic::render
