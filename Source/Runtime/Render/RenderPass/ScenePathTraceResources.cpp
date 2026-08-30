#include "Runtime/Render/RenderPass/ScenePathTraceResources.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"
#include "Runtime/Scene/SceneDocument.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

constexpr const char* kDefaultPathTraceScenePath = PROJECT_SOURCE_DIR "/Asset/meet_mat.glb";
constexpr int32_t kGltfTriangleListMode = 4;
constexpr uint32_t kInvalidMaterialTextureIndex = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kPrimitiveHasAuthoredTangents = 1u << 0u;

using SceneResourceLogClock = std::chrono::steady_clock;

double sceneResourceElapsedMilliseconds(SceneResourceLogClock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(SceneResourceLogClock::now() - begin).count();
}

class SceneResourceLogScope {
public:
    explicit SceneResourceLogScope(std::string label)
        : label_(std::move(label))
    {
        spdlog::info("[SceneResources] Begin {}", label_);
    }

    ~SceneResourceLogScope()
    {
        spdlog::info("[SceneResources] End {} in {:.2f} ms", label_, sceneResourceElapsedMilliseconds(begin_));
    }

private:
    std::string label_;
    SceneResourceLogClock::time_point begin_ = SceneResourceLogClock::now();
};

struct ScenePathTraceGpuVertex {
    float position[4] = {};
    float normal[4] = {};
    float tangent[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float texcoord[4] = {};
};

struct ScenePathTraceGpuPrimitive {
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t flags = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

struct ScenePathTraceGpuInstance {
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t flags = 0;
    float rayConeLodConstant = 0.0f;
};

struct ScenePathTraceGpuMaterial {
    float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float emissive[4] = {};
    float params[4] = {};
    float textureParams[4] = {1.0f, 1.0f, 0.0f, 0.0f};
    float glassParams[4] = {0.0f, 1.5f, 0.0f, 0.0f};
    float attenuationColor[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    float diffuseTransmission[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    float rtxcrHairBaseColor[4] = {0.2f, 0.2f, 0.2f, 0.0f};
    float rtxcrHairParams0[4] = {0.3f, 0.3f, 1.55f, 3.0f};
    float rtxcrHairParams1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float rtxcrHairDiffuseTint[4] = {};
    struct TextureInfo {
        uint32_t textureIndex = kInvalidMaterialTextureIndex;
        uint32_t texCoord = 0;
        uint32_t ntcTextureSetIndex = kInvalidNeuralTextureSetIndex;
        uint32_t ntcChannelMapping = UINT32_MAX;
        float transform0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float transform1[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    };
    TextureInfo baseColorTexture;
    TextureInfo metallicRoughnessTexture;
    TextureInfo normalTexture;
    TextureInfo occlusionTexture;
    TextureInfo emissiveTexture;
    TextureInfo transmissionTexture;
    TextureInfo thicknessTexture;
    TextureInfo diffuseTransmissionTexture;
    TextureInfo diffuseTransmissionColorTexture;
};

struct ScenePathTraceGpuScene {
    std::vector<ScenePathTraceGpuVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<ScenePathTraceGpuPrimitive> primitives;
    std::vector<ScenePathTraceGpuInstance> instances;
    std::vector<ScenePathTraceGpuMaterial> materials;
};

struct ScenePathTraceTextureMipUpload {
    uint64_t bufferOffset = 0;
    uint32_t width = 1;
    uint32_t height = 1;
    uint64_t byteSize = 4;
};

struct ScenePathTraceMaterialTexture {
    std::vector<ScenePathTraceTextureMipUpload> mipUploads;
    std::shared_ptr<Buffer> uploadBuffer;
    uint64_t uploadBufferOffset = 0;
    uint64_t uploadAllocationSize = 0;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t mipCount = 1;
    uint64_t byteSize = 4;
    Format format = Format::Rgba8Unorm;
    ResourceState state = ResourceState::Undefined;
    bool uploaded = false;
};

struct DecodedMaterialTexture {
    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    std::string label;
    const std::vector<scene::RenderImage::Mip>* preparedMips = nullptr;
};

struct ScenePathTraceBufferUpload {
    std::shared_ptr<Buffer> stagingBuffer;
    Buffer* destination = nullptr;
    uint64_t sourceOffset = 0;
    uint64_t byteSize = 0;
};


std::string resultMessage(std::string_view label, const Result& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

class SceneUploadStagingArena {
public:
    static constexpr uint64_t kPageByteSize = 64ull * 1024ull * 1024ull;

    ~SceneUploadStagingArena()
    {
        clear();
    }

    Result upload(
        Device& device,
        const void* data,
        uint64_t byteSize,
        uint64_t alignment,
        std::shared_ptr<Buffer>& outBuffer,
        uint64_t& outOffset,
        std::string& log,
        std::string_view label)
    {
        if (data == nullptr || byteSize == 0) {
            return makeError(Error::InvalidArgument);
        }

        alignment = std::max<uint64_t>(alignment, 1ull);
        Page* selectedPage = nullptr;
        uint64_t selectedOffset = 0;
        for (const std::unique_ptr<Page>& page : pages_) {
            const uint64_t alignedOffset = alignUp(page->cursor, alignment);
            if (alignedOffset <= page->capacity &&
                byteSize <= page->capacity - alignedOffset) {
                selectedPage = page.get();
                selectedOffset = alignedOffset;
                break;
            }
        }

        if (selectedPage == nullptr) {
            const uint64_t pageSize = std::max(kPageByteSize, alignUp(byteSize, alignment));
            std::unique_ptr<Buffer> buffer;
            Result result = device.createBuffer(
                BufferDesc{
                    .size = pageSize,
                    .usage = BufferUsageBits::TransferSource,
                    .memoryLocation = MemoryLocation::HostUpload,
                    .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy,
                },
                buffer);
            if (!result || buffer == nullptr) {
                log += resultMessage(std::string("createBuffer(") + std::string(label) + ")", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }

            auto page = std::make_unique<Page>();
            page->buffer = std::shared_ptr<Buffer>(std::move(buffer));
            page->capacity = pageSize;
            page->mapped = page->buffer->map();
            if (page->mapped == nullptr) {
                log = std::string(label) + " failed to map staging page";
                return makeError(Error::Failure);
            }
            selectedPage = page.get();
            selectedOffset = 0;
            pages_.push_back(std::move(page));
        }

        std::memcpy(
            static_cast<uint8_t*>(selectedPage->mapped) + selectedOffset,
            data,
            static_cast<size_t>(byteSize));
        selectedPage->buffer->flush(selectedOffset, byteSize);
        selectedPage->cursor = selectedOffset + byteSize;
        outBuffer = selectedPage->buffer;
        outOffset = selectedOffset;
        return {};
    }

    void reset()
    {
        for (const std::unique_ptr<Page>& page : pages_) {
            page->cursor = 0;
        }
    }

    void clear()
    {
        pages_.clear();
    }

private:
    struct Page {
        ~Page()
        {
            if (mapped != nullptr && buffer != nullptr) {
                buffer->unmap();
            }
        }

        std::shared_ptr<Buffer> buffer;
        void* mapped = nullptr;
        uint64_t capacity = 0;
        uint64_t cursor = 0;
    };

    static uint64_t alignUp(uint64_t value, uint64_t alignment)
    {
        const uint64_t remainder = value % alignment;
        if (remainder == 0) {
            return value;
        }
        return value + alignment - remainder;
    }

    std::vector<std::unique_ptr<Page>> pages_;
};

std::filesystem::path scenePathFromProperties(const RenderGraphProperties& props)
{
    if (props.contains("path") && props["path"].is_string()) {
        std::filesystem::path path = props["path"].get<std::string>();
        if (path.is_relative()) {
            path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
        }
        return path;
    }
    return kDefaultPathTraceScenePath;
}

void appendScenePathTraceWarning(std::string& log, std::string_view message)
{
    if (!log.empty() && log.back() != '\n') {
        log += '\n';
    }
    log += "Warning: ";
    log += message;
    log += '\n';
}

void appendLogBlock(std::string& log, const std::string& message)
{
    if (message.empty()) {
        return;
    }
    if (!log.empty() && log.back() != '\n') {
        log += '\n';
    }
    log += message;
    if (log.back() != '\n') {
        log += '\n';
    }
}

Result uploadStorageBuffer(
    Device& device,
    const void* data,
    uint64_t byteSize,
    uint32_t structureStride,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log,
    std::string_view label,
    std::vector<ScenePathTraceBufferUpload>* pendingUploads = nullptr,
    SceneUploadStagingArena* stagingArena = nullptr)
{
    if (data == nullptr || byteSize == 0) {
        log = std::string(label) + " upload data is empty";
        return makeError(Error::InvalidArgument);
    }

    const bool deviceLocal = pendingUploads != nullptr;
    Result result = device.createBuffer(
        BufferDesc{
            .size = byteSize,
            .structureStride = structureStride,
            .usage = deviceLocal
                ? BufferUsageBits::Storage | BufferUsageBits::TransferDestination
                : BufferUsageBits::Storage,
            .memoryLocation = deviceLocal ? MemoryLocation::Device : MemoryLocation::HostUpload,
            .queueAccess = deviceLocal
                ? QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy
                : QueueAccessBits::Graphics,
        },
        outBuffer);
    if (!result || outBuffer == nullptr) {
        log += resultMessage(std::string("createBuffer(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }

    std::shared_ptr<Buffer> stagingBuffer;
    uint64_t stagingOffset = 0;
    Buffer* uploadBuffer = outBuffer.get();
    if (deviceLocal) {
        if (stagingArena == nullptr) {
            log = std::string(label) + " requires a staging arena for a device-local upload";
            return makeError(Error::InvalidArgument);
        }
        result = stagingArena->upload(
            device,
            data,
            byteSize,
            16,
            stagingBuffer,
            stagingOffset,
            log,
            label);
        if (!result) {
            return result;
        }
        uploadBuffer = stagingBuffer.get();
    }

    if (!deviceLocal) {
        void* mapped = uploadBuffer->map();
        if (mapped == nullptr) {
            log = std::string(label) + " failed to map upload buffer";
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, data, static_cast<size_t>(byteSize));
        uploadBuffer->flush(0, byteSize);
        uploadBuffer->unmap();
    }
    if (deviceLocal) {
        pendingUploads->push_back(ScenePathTraceBufferUpload{
            .stagingBuffer = stagingBuffer,
            .destination = outBuffer.get(),
            .sourceOffset = stagingOffset,
            .byteSize = byteSize,
        });
    }
    return {};
}

Result createTextureUploadBuffer(
    Device& device,
    SceneUploadStagingArena& stagingArena,
    const void* data,
    uint64_t byteSize,
    uint64_t alignment,
    std::shared_ptr<Buffer>& outBuffer,
    uint64_t& outOffset,
    std::string& log,
    std::string_view label)
{
    return stagingArena.upload(
        device,
        data,
        byteSize,
        alignment,
        outBuffer,
        outOffset,
        log,
        label);
}

uint32_t mipCountForDimensions(uint32_t width, uint32_t height)
{
    uint32_t mipCount = 1;
    while (width > 1 || height > 1) {
        width = std::max(width / 2u, 1u);
        height = std::max(height / 2u, 1u);
        ++mipCount;
    }
    return mipCount;
}

uint64_t rgba8ByteSize(uint32_t width, uint32_t height)
{
    return static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ull;
}

std::vector<uint8_t> buildNextRgba8Mip(
    const uint8_t* sourcePixels,
    uint32_t sourceWidth,
    uint32_t sourceHeight)
{
    const uint32_t targetWidth = std::max(sourceWidth / 2u, 1u);
    const uint32_t targetHeight = std::max(sourceHeight / 2u, 1u);
    std::vector<uint8_t> target(static_cast<size_t>(rgba8ByteSize(targetWidth, targetHeight)));

    for (uint32_t y = 0; y < targetHeight; ++y) {
        for (uint32_t x = 0; x < targetWidth; ++x) {
            uint32_t sums[4] = {};
            uint32_t sampleCount = 0;
            for (uint32_t offsetY = 0; offsetY < 2; ++offsetY) {
                const uint32_t sourceY = std::min(y * 2u + offsetY, sourceHeight - 1u);
                for (uint32_t offsetX = 0; offsetX < 2; ++offsetX) {
                    const uint32_t sourceX = std::min(x * 2u + offsetX, sourceWidth - 1u);
                    const size_t sourceOffset =
                        static_cast<size_t>(sourceY * sourceWidth + sourceX) * 4u;
                    for (uint32_t component = 0; component < 4; ++component) {
                        sums[component] += sourcePixels[sourceOffset + component];
                    }
                    ++sampleCount;
                }
            }

            const size_t targetOffset = static_cast<size_t>(y * targetWidth + x) * 4u;
            for (uint32_t component = 0; component < 4; ++component) {
                target[targetOffset + component] =
                    static_cast<uint8_t>((sums[component] + sampleCount / 2u) / sampleCount);
            }
        }
    }
    return target;
}

std::vector<DecodedMaterialTexture> buildMaterialMipChain(
    const uint8_t* pixels,
    uint32_t width,
    uint32_t height,
    std::string_view label)
{
    std::vector<DecodedMaterialTexture> mipChain;
    mipChain.reserve(mipCountForDimensions(width, height));

    DecodedMaterialTexture baseMip;
    baseMip.width = width;
    baseMip.height = height;
    baseMip.label = std::string(label);
    const uint64_t baseByteSize = rgba8ByteSize(width, height);
    baseMip.pixels.assign(pixels, pixels + static_cast<size_t>(baseByteSize));
    mipChain.push_back(std::move(baseMip));

    while (mipChain.back().width > 1 || mipChain.back().height > 1) {
        const DecodedMaterialTexture& sourceMip = mipChain.back();
        DecodedMaterialTexture nextMip;
        nextMip.width = std::max(sourceMip.width / 2u, 1u);
        nextMip.height = std::max(sourceMip.height / 2u, 1u);
        nextMip.label = std::string(label);
        nextMip.pixels = buildNextRgba8Mip(
            sourceMip.pixels.data(),
            sourceMip.width,
            sourceMip.height);
        mipChain.push_back(std::move(nextMip));
    }
    return mipChain;
}

bool decodeSceneTexture(
    const scene::Scene& loadedScene,
    uint32_t textureIndex,
    DecodedMaterialTexture& outTexture,
    std::string& log)
{
    outTexture = DecodedMaterialTexture{};
    if (textureIndex >= loadedScene.textures().size()) {
        return false;
    }

    const scene::RenderTexture& texture = loadedScene.textures()[textureIndex];
    if (texture.imageIndex < 0 || static_cast<size_t>(texture.imageIndex) >= loadedScene.images().size()) {
        return false;
    }

    const scene::RenderImage& image = loadedScene.images()[static_cast<size_t>(texture.imageIndex)];
    outTexture.label = texture.name.empty() ? image.name : texture.name;
    if (!image.decodedMips.empty()) {
        outTexture.width = image.decodedMips.front().width;
        outTexture.height = image.decodedMips.front().height;
        outTexture.preparedMips = &image.decodedMips;
        return outTexture.width > 0 && outTexture.height > 0;
    }
    if (image.decodeAttempted) {
        appendScenePathTraceWarning(
            log,
            image.decodeWarning.empty() ? "failed to decode material texture" : image.decodeWarning);
        return false;
    }

    int width = 0;
    int height = 0;
    int channelCount = 0;
    stbi_uc* pixels = nullptr;
    if (!image.encodedData.empty()) {
        if (image.encodedData.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            appendScenePathTraceWarning(log, "embedded glTF image is too large to decode");
            return false;
        }
        pixels = stbi_load_from_memory(
            image.encodedData.data(),
            static_cast<int>(image.encodedData.size()),
            &width,
            &height,
            &channelCount,
            4);
    } else if (!image.uri.empty()) {
        if (image.uri.rfind("data:", 0) == 0) {
            appendScenePathTraceWarning(log, "data URI material textures are not supported yet");
            return false;
        }
        std::filesystem::path imagePath = image.uri;
        if (imagePath.is_relative()) {
            imagePath = loadedScene.filename().parent_path() / imagePath;
        }
        pixels = stbi_load(imagePath.string().c_str(), &width, &height, &channelCount, 4);
    }

    if (pixels == nullptr || width <= 0 || height <= 0) {
        std::string message = "failed to decode material texture";
        if (!outTexture.label.empty()) {
            message += " '";
            message += outTexture.label;
            message += "'";
        }
        if (const char* reason = stbi_failure_reason()) {
            message += ": ";
            message += reason;
        }
        appendScenePathTraceWarning(log, message);
        return false;
    }

    const uint64_t byteSize = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ull;
    if (byteSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        stbi_image_free(pixels);
        appendScenePathTraceWarning(log, "decoded material texture is too large");
        return false;
    }
    outTexture.width = static_cast<uint32_t>(width);
    outTexture.height = static_cast<uint32_t>(height);
    outTexture.pixels.assign(pixels, pixels + static_cast<size_t>(byteSize));
    stbi_image_free(pixels);
    return true;
}

Result createMaterialTexture(
    Device& device,
    SceneUploadStagingArena& stagingArena,
    const uint8_t* pixels,
    uint32_t width,
    uint32_t height,
    std::string_view label,
    ScenePathTraceMaterialTexture& outTexture,
    std::string& log,
    const std::vector<scene::RenderImage::Mip>* preparedMips = nullptr)
{
    if ((pixels == nullptr && (preparedMips == nullptr || preparedMips->empty())) ||
        width == 0 || height == 0) {
        return makeError(Error::InvalidArgument);
    }

    outTexture = ScenePathTraceMaterialTexture{};
    outTexture.width = width;
    outTexture.height = height;
    outTexture.format = Format::Rgba8Unorm;
    std::vector<DecodedMaterialTexture> generatedMipChain;
    if (preparedMips == nullptr || preparedMips->empty()) {
        generatedMipChain = buildMaterialMipChain(pixels, width, height, label);
    }
    const size_t mipCount = preparedMips != nullptr && !preparedMips->empty()
        ? preparedMips->size()
        : generatedMipChain.size();
    outTexture.mipCount = static_cast<uint32_t>(mipCount);
    outTexture.mipUploads.reserve(mipCount);
    outTexture.byteSize = 0;
    const uint64_t uploadAlignment = std::max<uint64_t>(
        device.capabilities().textureUploadBufferOffsetAlignment,
        1ull);
    std::vector<uint8_t> packedMipData;
    for (uint32_t mipIndex = 0; mipIndex < mipCount; ++mipIndex) {
        const uint32_t mipWidth = preparedMips != nullptr && !preparedMips->empty()
            ? (*preparedMips)[mipIndex].width
            : generatedMipChain[mipIndex].width;
        const uint32_t mipHeight = preparedMips != nullptr && !preparedMips->empty()
            ? (*preparedMips)[mipIndex].height
            : generatedMipChain[mipIndex].height;
        const std::vector<uint8_t>& mipPixels = preparedMips != nullptr && !preparedMips->empty()
            ? (*preparedMips)[mipIndex].pixels
            : generatedMipChain[mipIndex].pixels;
        ScenePathTraceTextureMipUpload upload;
        upload.width = mipWidth;
        upload.height = mipHeight;
        upload.byteSize = rgba8ByteSize(mipWidth, mipHeight);
        upload.bufferOffset = (packedMipData.size() + uploadAlignment - 1u) /
            uploadAlignment * uploadAlignment;
        if (mipPixels.size() < upload.byteSize ||
            upload.bufferOffset > std::numeric_limits<size_t>::max() - upload.byteSize) {
            return makeError(Error::InvalidArgument);
        }
        packedMipData.resize(static_cast<size_t>(upload.bufferOffset + upload.byteSize));
        std::memcpy(
            packedMipData.data() + upload.bufferOffset,
            mipPixels.data(),
            static_cast<size_t>(upload.byteSize));
        outTexture.byteSize += upload.byteSize;
        outTexture.mipUploads.push_back(std::move(upload));
    }

    std::string uploadLabel = "ScenePathTracePass packed texture upload ";
    uploadLabel += std::string(label);
    Result result = createTextureUploadBuffer(
        device,
        stagingArena,
        packedMipData.data(),
        packedMipData.size(),
        uploadAlignment,
        outTexture.uploadBuffer,
        outTexture.uploadBufferOffset,
        log,
        uploadLabel);
    if (!result) {
        return result;
    }
    outTexture.uploadAllocationSize = packedMipData.size();

    result = device.createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
            .format = outTexture.format,
            .width = width,
            .height = height,
            .depth = 1,
            .mipCount = outTexture.mipCount,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy,
        },
        outTexture.texture);
    if (!result || outTexture.texture == nullptr) {
        log += resultMessage(std::string("createTexture(ScenePathTracePass material texture ") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }

    result = device.createTextureView(
        *outTexture.texture,
        TextureViewDesc{
            .format = outTexture.format,
            .baseMip = 0,
            .mipCount = outTexture.mipCount,
            .baseLayer = 0,
            .layerCount = 1,
        },
        outTexture.view);
    if (!result || outTexture.view == nullptr) {
        log += resultMessage(std::string("createTextureView(ScenePathTracePass material texture ") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

uint32_t materialTextureIndex(
    int32_t textureIndex,
    const std::vector<uint32_t>& textureIndexMap)
{
    if (textureIndex < 0 || static_cast<size_t>(textureIndex) >= textureIndexMap.size()) {
        return kInvalidMaterialTextureIndex;
    }
    return textureIndexMap[static_cast<size_t>(textureIndex)];
}

ScenePathTraceGpuMaterial::TextureInfo makeGpuTextureInfo(
    const scene::RenderTextureInfo& textureInfo,
    const scene::Scene& loadedScene,
    const std::vector<uint32_t>& textureIndexMap,
    const std::vector<uint32_t>& neuralTextureSetIndexMap,
    std::string& log,
    std::string_view textureLabel)
{
    ScenePathTraceGpuMaterial::TextureInfo gpuTextureInfo;
    gpuTextureInfo.textureIndex = materialTextureIndex(textureInfo.textureIndex, textureIndexMap);
    if (textureInfo.textureIndex >= 0 &&
        static_cast<size_t>(textureInfo.textureIndex) < loadedScene.textures().size() &&
        static_cast<size_t>(textureInfo.textureIndex) < neuralTextureSetIndexMap.size()) {
        const scene::RenderTexture& logicalTexture =
            loadedScene.textures()[static_cast<size_t>(textureInfo.textureIndex)];
        gpuTextureInfo.ntcTextureSetIndex =
            neuralTextureSetIndexMap[static_cast<size_t>(textureInfo.textureIndex)];
        if (gpuTextureInfo.ntcTextureSetIndex != kInvalidNeuralTextureSetIndex) {
            gpuTextureInfo.ntcChannelMapping = 0xffffffffu;
            for (uint32_t channelIndex = 0; channelIndex < logicalTexture.ntcChannelCount;
                 ++channelIndex) {
                const uint32_t channel =
                    static_cast<uint8_t>(logicalTexture.ntcChannels[channelIndex]);
                gpuTextureInfo.ntcChannelMapping &= ~(0xffu << (channelIndex * 8u));
                gpuTextureInfo.ntcChannelMapping |= channel << (channelIndex * 8u);
            }
            gpuTextureInfo.textureIndex = kInvalidMaterialTextureIndex;
        }
    }
    if (textureInfo.texCoord > 0) {
        if (gpuTextureInfo.textureIndex != kInvalidMaterialTextureIndex) {
            appendScenePathTraceWarning(
                log,
                std::string(textureLabel) + " requests TEXCOORD_" +
                    std::to_string(textureInfo.texCoord) +
                    "; ScenePathTracePass currently samples TEXCOORD_0");
        }
        gpuTextureInfo.texCoord = 0;
    }
    gpuTextureInfo.transform0[0] = textureInfo.uvTransform[0];
    gpuTextureInfo.transform0[1] = textureInfo.uvTransform[1];
    gpuTextureInfo.transform0[2] = textureInfo.uvTransform[2];
    gpuTextureInfo.transform1[0] = textureInfo.uvTransform[3];
    gpuTextureInfo.transform1[1] = textureInfo.uvTransform[4];
    gpuTextureInfo.transform1[2] = textureInfo.uvTransform[5];
    return gpuTextureInfo;
}

float alphaModeCode(const std::string& alphaMode)
{
    if (alphaMode == "MASK") {
        return 1.0f;
    }
    if (alphaMode == "BLEND") {
        return 0.0f;
    }
    return 0.0f;
}

ScenePathTraceGpuMaterial makeMaterial(
    const scene::RenderMaterial& material,
    const scene::Scene& loadedScene,
    const std::vector<uint32_t>& textureIndexMap,
    const std::vector<uint32_t>& neuralTextureSetIndexMap,
    std::string& log)
{
    ScenePathTraceGpuMaterial gpuMaterial;
    gpuMaterial.baseColor[0] = material.baseColorFactor.x;
    gpuMaterial.baseColor[1] = material.baseColorFactor.y;
    gpuMaterial.baseColor[2] = material.baseColorFactor.z;
    gpuMaterial.baseColor[3] = material.baseColorFactor.w;
    gpuMaterial.emissive[0] = material.emissiveFactor.x;
    gpuMaterial.emissive[1] = material.emissiveFactor.y;
    gpuMaterial.emissive[2] = material.emissiveFactor.z;
    gpuMaterial.emissive[3] = 0.0f;
    gpuMaterial.params[0] = material.metallicFactor;
    gpuMaterial.params[1] = material.roughnessFactor;
    gpuMaterial.params[2] = material.alphaCutoff;
    gpuMaterial.params[3] = material.doubleSided ? 1.0f : 0.0f;
    gpuMaterial.textureParams[0] = material.normalTextureScale;
    gpuMaterial.textureParams[1] = material.occlusionTextureStrength;
    gpuMaterial.textureParams[2] = 0.0f;
    gpuMaterial.textureParams[3] = alphaModeCode(material.alphaMode);
    if (material.alphaMode == "BLEND") {
        std::string message =
            "alphaMode BLEND is not supported by ScenePathTracePass yet; rendering as OPAQUE";
        if (!material.name.empty()) {
            message += " for material '";
            message += material.name;
            message += "'";
        }
        appendScenePathTraceWarning(log, message);
    }
    gpuMaterial.glassParams[0] = material.transmissionFactor;
    gpuMaterial.glassParams[1] = material.ior;
    gpuMaterial.glassParams[2] = material.thicknessFactor;
    gpuMaterial.glassParams[3] = material.attenuationDistance;
    gpuMaterial.attenuationColor[0] = material.attenuationColor.x;
    gpuMaterial.attenuationColor[1] = material.attenuationColor.y;
    gpuMaterial.attenuationColor[2] = material.attenuationColor.z;
    gpuMaterial.attenuationColor[3] = 0.0f;
    gpuMaterial.diffuseTransmission[0] = material.diffuseTransmissionColor.x;
    gpuMaterial.diffuseTransmission[1] = material.diffuseTransmissionColor.y;
    gpuMaterial.diffuseTransmission[2] = material.diffuseTransmissionColor.z;
    gpuMaterial.diffuseTransmission[3] = material.diffuseTransmissionFactor;
    gpuMaterial.rtxcrHairBaseColor[0] = material.rtxcrHairBaseColor.x;
    gpuMaterial.rtxcrHairBaseColor[1] = material.rtxcrHairBaseColor.y;
    gpuMaterial.rtxcrHairBaseColor[2] = material.rtxcrHairBaseColor.z;
    gpuMaterial.rtxcrHairBaseColor[3] = material.rtxcrHair ? 1.0f : 0.0f;
    gpuMaterial.rtxcrHairParams0[0] = material.rtxcrHairLongitudinalRoughness;
    gpuMaterial.rtxcrHairParams0[1] = material.rtxcrHairAzimuthalRoughness;
    gpuMaterial.rtxcrHairParams0[2] = material.rtxcrHairIor;
    gpuMaterial.rtxcrHairParams0[3] = material.rtxcrHairCuticleAngleDegrees;
    gpuMaterial.rtxcrHairParams1[0] = material.rtxcrHairMelanin;
    gpuMaterial.rtxcrHairParams1[1] = material.rtxcrHairMelaninRedness;
    gpuMaterial.rtxcrHairParams1[2] = material.rtxcrHairDiffuseReflectionWeight;
    gpuMaterial.rtxcrHairDiffuseTint[0] = material.rtxcrHairDiffuseReflectionTint.x;
    gpuMaterial.rtxcrHairDiffuseTint[1] = material.rtxcrHairDiffuseReflectionTint.y;
    gpuMaterial.rtxcrHairDiffuseTint[2] = material.rtxcrHairDiffuseReflectionTint.z;
    const auto makeTextureInfo = [&](const scene::RenderTextureInfo& info,
                                     std::string_view label) {
        return makeGpuTextureInfo(
            info,
            loadedScene,
            textureIndexMap,
            neuralTextureSetIndexMap,
            log,
            label);
    };
    gpuMaterial.baseColorTexture = makeTextureInfo(material.baseColorTexture, "baseColorTexture");
    gpuMaterial.metallicRoughnessTexture = makeGpuTextureInfo(
        material.metallicRoughnessTexture,
        loadedScene,
        textureIndexMap,
        neuralTextureSetIndexMap,
        log,
        "metallicRoughnessTexture");
    gpuMaterial.normalTexture = makeTextureInfo(material.normalTexture, "normalTexture");
    gpuMaterial.occlusionTexture = makeTextureInfo(material.occlusionTexture, "occlusionTexture");
    gpuMaterial.emissiveTexture = makeTextureInfo(material.emissiveTexture, "emissiveTexture");
    gpuMaterial.transmissionTexture = makeTextureInfo(material.transmissionTexture, "transmissionTexture");
    gpuMaterial.thicknessTexture = makeTextureInfo(material.thicknessTexture, "thicknessTexture");
    gpuMaterial.diffuseTransmissionTexture = makeTextureInfo(
        material.diffuseTransmissionTexture,
        "diffuseTransmissionTexture");
    gpuMaterial.diffuseTransmissionColorTexture = makeTextureInfo(
        material.diffuseTransmissionColorTexture,
        "diffuseTransmissionColorTexture");
    return gpuMaterial;
}

uint32_t materialIndexForNode(const scene::RenderNode& renderNode, uint32_t materialCount)
{
    if (renderNode.materialIndex >= 0 &&
        static_cast<uint32_t>(renderNode.materialIndex) < materialCount) {
        return static_cast<uint32_t>(renderNode.materialIndex);
    }
    return 0;
}

float safeLog2(float value)
{
    return std::log2(std::max(value, 0.0000001f));
}

float3 transformPointForLod(const float4x4& matrix, const float3& point)
{
    return matrix * point;
}

bool primitiveTriangleVertexIndex(
    const scene::RenderPrimitive& primitive,
    uint64_t sourceIndex,
    uint32_t& outVertexIndex)
{
    if (primitive.indices.empty()) {
        if (sourceIndex >= primitive.positions.size()) {
            return false;
        }
        outVertexIndex = static_cast<uint32_t>(sourceIndex);
        return true;
    }

    if (sourceIndex >= primitive.indices.size()) {
        return false;
    }
    const uint32_t vertexIndex = primitive.indices[static_cast<size_t>(sourceIndex)];
    if (vertexIndex >= primitive.positions.size()) {
        return false;
    }
    outVertexIndex = vertexIndex;
    return true;
}

float rayConeLodConstantForPrimitive(
    const scene::RenderPrimitive& primitive,
    const float4x4& worldMatrix)
{
    const uint64_t sourceIndexCount = primitive.indices.empty()
        ? (primitive.positions.size() / 3) * 3
        : (primitive.indices.size() / 3) * 3;
    if (primitive.mode != kGltfTriangleListMode ||
        primitive.positions.size() < 3 ||
        primitive.texcoords0.empty() ||
        sourceIndexCount < 3) {
        return 0.0f;
    }

    double weightedLod = 0.0;
    double totalWeight = 0.0;
    for (uint64_t sourceIndex = 0; sourceIndex + 2 < sourceIndexCount; sourceIndex += 3) {
        uint32_t i0 = 0;
        uint32_t i1 = 0;
        uint32_t i2 = 0;
        if (!primitiveTriangleVertexIndex(primitive, sourceIndex + 0, i0) ||
            !primitiveTriangleVertexIndex(primitive, sourceIndex + 1, i1) ||
            !primitiveTriangleVertexIndex(primitive, sourceIndex + 2, i2) ||
            i0 >= primitive.texcoords0.size() ||
            i1 >= primitive.texcoords0.size() ||
            i2 >= primitive.texcoords0.size()) {
            continue;
        }

        const float3 p0 = transformPointForLod(worldMatrix, primitive.positions[i0]);
        const float3 p1 = transformPointForLod(worldMatrix, primitive.positions[i1]);
        const float3 p2 = transformPointForLod(worldMatrix, primitive.positions[i2]);
        const float2 uv0 = primitive.texcoords0[i0];
        const float2 uv1 = primitive.texcoords0[i1];
        const float2 uv2 = primitive.texcoords0[i2];

        const float3 worldEdge0 = p1 - p0;
        const float3 worldEdge1 = p2 - p0;
        const float worldArea2 = length(cross(worldEdge0, worldEdge1));
        const float2 uvEdge0 = uv1 - uv0;
        const float2 uvEdge1 = uv2 - uv0;
        const float texcoordArea2 = std::abs(uvEdge0.x * uvEdge1.y - uvEdge0.y * uvEdge1.x);
        if (worldArea2 <= 0.0000001f || texcoordArea2 <= 0.0000001f) {
            continue;
        }

        const float lodConstant = 0.5f * safeLog2(texcoordArea2 / worldArea2);
        if (!std::isfinite(lodConstant)) {
            continue;
        }
        weightedLod += static_cast<double>(lodConstant) * static_cast<double>(worldArea2);
        totalWeight += static_cast<double>(worldArea2);
    }

    return totalWeight > 0.0 ? static_cast<float>(weightedLod / totalWeight) : 0.0f;
}

bool appendPrimitiveGeometry(
    const scene::RenderPrimitive& primitive,
    ScenePathTraceGpuScene& outScene,
    ScenePathTraceGpuPrimitive& outPrimitive)
{
    const uint64_t sourceIndexCount = primitive.indices.empty()
        ? (primitive.positions.size() / 3) * 3
        : (primitive.indices.size() / 3) * 3;
    if (primitive.mode != kGltfTriangleListMode ||
        primitive.positions.size() < 3 ||
        sourceIndexCount < 3 ||
        sourceIndexCount > std::numeric_limits<uint32_t>::max() ||
        primitive.positions.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    outPrimitive = ScenePathTraceGpuPrimitive{
        .firstVertex = static_cast<uint32_t>(outScene.vertices.size()),
        .vertexCount = static_cast<uint32_t>(primitive.positions.size()),
        .firstIndex = static_cast<uint32_t>(outScene.indices.size()),
        .indexCount = static_cast<uint32_t>(sourceIndexCount),
        .flags = primitive.hasAuthoredTangents ? kPrimitiveHasAuthoredTangents : 0u,
    };

    for (size_t vertexIndex = 0; vertexIndex < primitive.positions.size(); ++vertexIndex) {
        const float3 position = primitive.positions[vertexIndex];
        const float3 normal = vertexIndex < primitive.normals.size()
            ? primitive.normals[vertexIndex]
            : float3(0.0f, 0.0f, 0.0f);
        const float4 tangent = vertexIndex < primitive.tangents.size()
            ? primitive.tangents[vertexIndex]
            : float4(1.0f, 0.0f, 0.0f, 1.0f);
        const float2 texcoord = vertexIndex < primitive.texcoords0.size()
            ? primitive.texcoords0[vertexIndex]
            : float2(0.0f, 0.0f);
        ScenePathTraceGpuVertex vertex;
        vertex.position[0] = position.x;
        vertex.position[1] = position.y;
        vertex.position[2] = position.z;
        vertex.position[3] = 1.0f;
        vertex.normal[0] = normal.x;
        vertex.normal[1] = normal.y;
        vertex.normal[2] = normal.z;
        vertex.normal[3] = 0.0f;
        vertex.tangent[0] = tangent.x;
        vertex.tangent[1] = tangent.y;
        vertex.tangent[2] = tangent.z;
        vertex.tangent[3] = tangent.w >= 0.0f ? 1.0f : -1.0f;
        vertex.texcoord[0] = texcoord.x;
        vertex.texcoord[1] = texcoord.y;
        outScene.vertices.push_back(vertex);
    }

    if (primitive.indices.empty()) {
        for (uint32_t index = 0; index < outPrimitive.indexCount; ++index) {
            outScene.indices.push_back(index);
        }
        return true;
    }

    for (uint32_t index = 0; index < outPrimitive.indexCount; ++index) {
        const uint32_t sourceIndex = primitive.indices[index];
        if (sourceIndex >= outPrimitive.vertexCount) {
            outScene.vertices.resize(outPrimitive.firstVertex);
            outScene.indices.resize(outPrimitive.firstIndex);
            return false;
        }
        outScene.indices.push_back(sourceIndex);
    }
    return true;
}

bool buildGpuScene(
    const scene::Scene& loadedScene,
    const std::vector<uint32_t>& textureIndexMap,
    const std::vector<uint32_t>& neuralTextureSetIndexMap,
    ScenePathTraceGpuScene& outScene,
    std::string& log)
{
    outScene = ScenePathTraceGpuScene{};
    outScene.materials.reserve(std::max<size_t>(loadedScene.materials().size(), 1));
    if (loadedScene.materials().empty()) {
        outScene.materials.push_back(ScenePathTraceGpuMaterial{});
    } else {
        for (const scene::RenderMaterial& material : loadedScene.materials()) {
            outScene.materials.push_back(makeMaterial(
                material,
                loadedScene,
                textureIndexMap,
                neuralTextureSetIndexMap,
                log));
        }
    }

    constexpr uint32_t kInvalidPrimitiveIndex = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> primitiveToGpuPrimitive(
        loadedScene.renderPrimitives().size(),
        kInvalidPrimitiveIndex);
    for (uint32_t primitiveIndex = 0; primitiveIndex < loadedScene.renderPrimitives().size(); ++primitiveIndex) {
        ScenePathTraceGpuPrimitive gpuPrimitive;
        if (!appendPrimitiveGeometry(loadedScene.renderPrimitives()[primitiveIndex], outScene, gpuPrimitive)) {
            continue;
        }
        primitiveToGpuPrimitive[primitiveIndex] = static_cast<uint32_t>(outScene.primitives.size());
        outScene.primitives.push_back(gpuPrimitive);
    }

    for (const scene::RenderNode& renderNode : loadedScene.renderNodes()) {
        if (!renderNode.visible ||
            renderNode.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(renderNode.renderPrimitiveIndex) >= primitiveToGpuPrimitive.size()) {
            continue;
        }
        const uint32_t primitiveIndex =
            primitiveToGpuPrimitive[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
        if (primitiveIndex == kInvalidPrimitiveIndex) {
            continue;
        }

        outScene.instances.push_back(ScenePathTraceGpuInstance{
            .primitiveIndex = primitiveIndex,
            .materialIndex = materialIndexForNode(
                renderNode,
                static_cast<uint32_t>(outScene.materials.size())),
            .flags = 0,
            .rayConeLodConstant = rayConeLodConstantForPrimitive(
                loadedScene.renderPrimitives()[static_cast<size_t>(renderNode.renderPrimitiveIndex)],
                renderNode.worldMatrix),
        });
    }

    if (outScene.vertices.empty() ||
        outScene.indices.empty() ||
        outScene.primitives.empty() ||
        outScene.materials.empty()) {
        log = "ScenePathTracePass found no triangle geometry suitable for path tracing";
        return false;
    }
    if (outScene.instances.empty()) {
        // Storage buffers cannot be empty, while an empty TLAS guarantees that
        // this placeholder is never addressed by a committed ray-query hit.
        outScene.instances.push_back(ScenePathTraceGpuInstance{});
    }
    return true;
}

std::vector<bool> referencedMaterialTextures(const scene::Scene& loadedScene)
{
    std::vector<bool> referenced(loadedScene.textures().size(), false);
    const auto mark = [&referenced](const scene::RenderTextureInfo& texture) {
        if (texture.textureIndex >= 0 &&
            static_cast<size_t>(texture.textureIndex) < referenced.size()) {
            referenced[static_cast<size_t>(texture.textureIndex)] = true;
        }
    };
    for (const scene::RenderMaterial& material : loadedScene.materials()) {
        mark(material.baseColorTexture);
        mark(material.metallicRoughnessTexture);
        mark(material.normalTexture);
        mark(material.occlusionTexture);
        mark(material.emissiveTexture);
        mark(material.transmissionTexture);
        mark(material.thicknessTexture);
        mark(material.diffuseTransmissionTexture);
        mark(material.diffuseTransmissionColorTexture);
    }
    return referenced;
}

} // namespace

struct ScenePathTraceResources::Impl {
    enum class AsyncPrepareStage : uint8_t {
        Idle,
        MaterialTextures,
        GpuPayload,
        AccelerationStructure,
        Buffers,
        SubmitPartialUploads,
        WaitForPartialUploads,
        SubmitUploads,
        WaitForGpu,
        Ready,
        Failed,
    };

    static constexpr uint64_t kUploadBatchByteLimit = 64ull * 1024ull * 1024ull;
    static constexpr uint32_t kUploadBatchRegionLimit = 128;

    uint64_t pendingUploadByteSize() const
    {
        uint64_t byteSize = neuralTextures.pendingUploadByteSize();
        for (const ScenePathTraceMaterialTexture& texture : materialTextures) {
            if (!texture.uploaded && texture.uploadBuffer != nullptr) {
                byteSize += texture.uploadAllocationSize;
            }
        }
        for (const ScenePathTraceBufferUpload& upload : bufferUploads) {
            byteSize += upload.byteSize;
        }
        return byteSize;
    }

    uint32_t pendingUploadRegionCount() const
    {
        uint64_t regionCount = bufferUploads.size() +
            neuralTextures.pendingUploadRegionCount();
        for (const ScenePathTraceMaterialTexture& texture : materialTextures) {
            if (!texture.uploaded && texture.uploadBuffer != nullptr) {
                regionCount += texture.mipUploads.size();
            }
        }
        return static_cast<uint32_t>(std::min<uint64_t>(regionCount, UINT32_MAX));
    }

    uint64_t nextMaterialTextureUploadByteSize(const scene::Scene& loadedScene) const
    {
        if (asyncTextureCursor >= loadedScene.textures().size()) {
            return 0;
        }
        if (asyncTextureCursor >= asyncReferencedTextures.size() ||
            !asyncReferencedTextures[asyncTextureCursor]) {
            return 0;
        }
        if (neuralTextures.logicalTextureSetIndex(
                static_cast<uint32_t>(asyncTextureCursor)) !=
            kInvalidNeuralTextureSetIndex) {
            return 0;
        }
        const scene::RenderTexture& texture = loadedScene.textures()[asyncTextureCursor];
        if (texture.imageIndex < 0 ||
            static_cast<size_t>(texture.imageIndex) >= loadedScene.images().size()) {
            return 0;
        }
        const size_t imageIndex = static_cast<size_t>(texture.imageIndex);
        if (imageIndex < asyncImageTextureIndexMap.size() &&
            asyncImageTextureIndexMap[imageIndex] != kInvalidMaterialTextureIndex) {
            return 0;
        }
        uint64_t byteSize = 0;
        for (const scene::RenderImage::Mip& mip : loadedScene.images()[imageIndex].decodedMips) {
            byteSize += mip.pixels.size();
        }
        return byteSize;
    }

    uint32_t nextMaterialTextureUploadRegionCount(const scene::Scene& loadedScene) const
    {
        if (asyncTextureCursor >= loadedScene.textures().size()) {
            return 0;
        }
        if (asyncTextureCursor >= asyncReferencedTextures.size() ||
            !asyncReferencedTextures[asyncTextureCursor]) {
            return 0;
        }
        if (neuralTextures.logicalTextureSetIndex(
                static_cast<uint32_t>(asyncTextureCursor)) !=
            kInvalidNeuralTextureSetIndex) {
            return 0;
        }
        const scene::RenderTexture& texture = loadedScene.textures()[asyncTextureCursor];
        if (texture.imageIndex < 0 ||
            static_cast<size_t>(texture.imageIndex) >= loadedScene.images().size()) {
            return 0;
        }
        const size_t imageIndex = static_cast<size_t>(texture.imageIndex);
        if (imageIndex < asyncImageTextureIndexMap.size() &&
            asyncImageTextureIndexMap[imageIndex] != kInvalidMaterialTextureIndex) {
            return 0;
        }
        return static_cast<uint32_t>(std::min<size_t>(
            loadedScene.images()[imageIndex].decodedMips.size(),
            UINT32_MAX));
    }

    bool shouldFlushBefore(uint64_t additionalBytes, uint32_t additionalRegions) const
    {
        const uint64_t pendingBytes = pendingUploadByteSize();
        const uint32_t pendingRegions = pendingUploadRegionCount();
        if (pendingBytes == 0 && pendingRegions == 0) {
            return false;
        }
        const bool byteLimitExceeded = additionalBytes > kUploadBatchByteLimit ||
            pendingBytes > kUploadBatchByteLimit - additionalBytes;
        const bool regionLimitExceeded = additionalRegions > kUploadBatchRegionLimit ||
            pendingRegions > kUploadBatchRegionLimit - additionalRegions;
        return byteLimitExceeded || regionLimitExceeded;
    }

    bool uploadBatchLimitReached() const
    {
        return pendingUploadByteSize() >= kUploadBatchByteLimit ||
            pendingUploadRegionCount() >= kUploadBatchRegionLimit;
    }

    ~Impl()
    {
        clear();
    }

    Result submitTextureUploads(Device& device, Queue& graphicsQueue, std::string& log)
    {
        const uint64_t batchByteSize = pendingUploadByteSize();
        const uint32_t batchRegionCount = pendingUploadRegionCount();
        // vkCmdConvertCooperativeVectorMatrixNV requires a graphics or compute
        // capable command buffer, so keep CoopVec weight conversion off a
        // transfer-only queue.
        Queue* uploadQueue = neuralTextures.cooperativeVectorActive()
            ? &graphicsQueue
            : device.getQueue(QueueType::Copy);
        if (uploadQueue == nullptr) {
            uploadQueue = &graphicsQueue;
        }
        const bool requiresGraphicsAcquire = uploadQueue != &graphicsQueue;

        Result result = device.createCommandPool(*uploadQueue, textureUploadCommandPool);
        if (!result || textureUploadCommandPool == nullptr) {
            log += resultMessage("createCommandPool(scene texture uploads)", result);
            return result ? makeError(Error::Failure) : result;
        }
        result = textureUploadCommandPool->createCommandBuffer(textureUploadCommandBuffer);
        if (!result || textureUploadCommandBuffer == nullptr) {
            log += resultMessage("createCommandBuffer(scene texture uploads)", result);
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createSemaphore(
            SemaphoreDesc{.initialValue = 0},
            textureUploadTimeline);
        if (!result || textureUploadTimeline == nullptr) {
            log += resultMessage("createSemaphore(scene uploads)", result);
            return result ? makeError(Error::Failure) : result;
        }
        result = textureUploadCommandBuffer->begin();
        if (!result) {
            log += resultMessage("CommandBuffer::begin(scene texture uploads)", result);
            return result;
        }
        for (ScenePathTraceMaterialTexture& texture : materialTextures) {
            result = uploadTexture(*textureUploadCommandBuffer, texture);
            if (!result) {
                return result;
            }
        }
        result = neuralTextures.recordUploads(*textureUploadCommandBuffer);
        if (!result) {
            return result;
        }
        for (const ScenePathTraceBufferUpload& upload : bufferUploads) {
            textureUploadCommandBuffer->copyBuffer(BufferCopyDesc{
                .source = upload.stagingBuffer.get(),
                .destination = upload.destination,
                .sourceOffset = upload.sourceOffset,
                .size = upload.byteSize,
            });
        }
        if (!requiresGraphicsAcquire) {
            transitionUploadsForRendering(*textureUploadCommandBuffer);
        }
        result = textureUploadCommandBuffer->end();
        if (!result) {
            log += resultMessage("CommandBuffer::end(scene texture uploads)", result);
            return result;
        }

        CommandBuffer* commandBuffers[] = {textureUploadCommandBuffer.get()};
        const SemaphoreSubmitDesc signal{
            .semaphore = textureUploadTimeline.get(),
            .value = 1,
            .stages = requiresGraphicsAcquire ? PipelineStageBits::Transfer : PipelineStageBits::AllCommands,
        };
        result = uploadQueue->submit(QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalSemaphores = &signal,
            .signalSemaphoreCount = 1,
        });
        if (!result) {
            log += resultMessage("Queue::submit(scene texture uploads)", result);
            return result;
        }

        textureUploadTimelineValue = 1;
        textureUploadsSubmitted = true;
        if (requiresGraphicsAcquire) {
            result = device.createCommandPool(graphicsQueue, textureAcquireCommandPool);
            if (!result || textureAcquireCommandPool == nullptr) {
                log += resultMessage("createCommandPool(scene upload acquire)", result);
                return result ? makeError(Error::Failure) : result;
            }
            result = textureAcquireCommandPool->createCommandBuffer(textureAcquireCommandBuffer);
            if (!result || textureAcquireCommandBuffer == nullptr) {
                log += resultMessage("createCommandBuffer(scene upload acquire)", result);
                return result ? makeError(Error::Failure) : result;
            }
            result = textureAcquireCommandBuffer->begin();
            if (!result) {
                log += resultMessage("CommandBuffer::begin(scene upload acquire)", result);
                return result;
            }
            transitionUploadsForRendering(*textureAcquireCommandBuffer);
            result = textureAcquireCommandBuffer->end();
            if (!result) {
                log += resultMessage("CommandBuffer::end(scene upload acquire)", result);
                return result;
            }

            CommandBuffer* acquireCommandBuffers[] = {textureAcquireCommandBuffer.get()};
            const SemaphoreSubmitDesc wait{
                .semaphore = textureUploadTimeline.get(),
                .value = 1,
                .stages = PipelineStageBits::AllCommands,
            };
            const SemaphoreSubmitDesc acquireSignal{
                .semaphore = textureUploadTimeline.get(),
                .value = 2,
                .stages = PipelineStageBits::AllCommands,
            };
            result = graphicsQueue.submit(QueueSubmitDesc{
                .waitSemaphores = &wait,
                .waitSemaphoreCount = 1,
                .commandBuffers = acquireCommandBuffers,
                .commandBufferCount = 1,
                .signalSemaphores = &acquireSignal,
                .signalSemaphoreCount = 1,
            });
            if (!result) {
                log += resultMessage("Queue::submit(scene upload acquire)", result);
                return result;
            }
            textureUploadTimelineValue = 2;
        }
        spdlog::info(
            "[SceneResources] Submitted scene upload batch bytes={} regions={} queue={} independentCopy={}",
            batchByteSize,
            batchRegionCount,
            uploadQueue->type() == QueueType::Copy ? "copy" : "graphics",
            device.capabilities().independentCopyQueue);
        return {};
    }

    bool textureUploadsReady() const
    {
        return !textureUploadsSubmitted ||
            (textureUploadTimeline != nullptr &&
                textureUploadTimeline->currentValue() >= textureUploadTimelineValue);
    }

    void retireCompletedTextureUploads()
    {
        if (!textureUploadsReady()) {
            return;
        }
        for (ScenePathTraceMaterialTexture& texture : materialTextures) {
            texture.uploadBuffer.reset();
        }
        neuralTextures.releaseUploadBuffers();
        bufferUploads.clear();
        stagingArena.reset();
        textureUploadCommandBuffer.reset();
        textureUploadCommandPool.reset();
        textureAcquireCommandBuffer.reset();
        textureAcquireCommandPool.reset();
        textureUploadTimeline.reset();
        textureUploadsSubmitted = false;
    }

    void waitForTextureUploads()
    {
        if (textureUploadsSubmitted && textureUploadTimeline != nullptr && !textureUploadsReady()) {
            (void)textureUploadTimeline->wait(textureUploadTimelineValue);
        }
        retireCompletedTextureUploads();
    }

    Result buildMaterialTextures(
        Device& device,
        const scene::Scene& loadedScene,
        std::vector<uint32_t>& outTextureIndexMap,
        std::string& log)
    {
        Result result = neuralTextures.prepare(device, loadedScene, log);
        if (!result) {
            return result;
        }
        materialTextures.clear();
        materialTextureViews.fill(nullptr);
        materialTextureCount = 0;
        textureIndexMap.clear();
        outTextureIndexMap.assign(loadedScene.textures().size(), kInvalidMaterialTextureIndex);
        const std::vector<bool> referencedTextures = referencedMaterialTextures(loadedScene);

        const uint8_t fallbackPixels[4] = {255, 255, 255, 255};
        ScenePathTraceMaterialTexture fallbackTexture;
        const auto fallbackBegin = SceneResourceLogClock::now();
        result = createMaterialTexture(
            device,
            stagingArena,
            fallbackPixels,
            1,
            1,
            "fallback",
            fallbackTexture,
            log);
        if (!result) {
            return result;
        }
        materialTextures.push_back(std::move(fallbackTexture));
        spdlog::info(
            "[SceneResources] Material texture fallback prepared in {:.2f} ms",
            sceneResourceElapsedMilliseconds(fallbackBegin));

        uint32_t decodedTextureCount = 0;
        uint64_t decodedTextureBytes = 0;
        std::vector<uint32_t> imageTextureIndexMap(
            loadedScene.images().size(),
            kInvalidMaterialTextureIndex);
        for (uint32_t textureIndex = 0; textureIndex < loadedScene.textures().size(); ++textureIndex) {
            const scene::RenderTexture& logicalTexture = loadedScene.textures()[textureIndex];
            if (!referencedTextures[textureIndex]) {
                continue;
            }
            if (neuralTextures.logicalTextureSetIndex(textureIndex) !=
                kInvalidNeuralTextureSetIndex) {
                continue;
            }
            if (logicalTexture.imageIndex >= 0 &&
                static_cast<size_t>(logicalTexture.imageIndex) < imageTextureIndexMap.size()) {
                const uint32_t existingIndex = imageTextureIndexMap[static_cast<size_t>(logicalTexture.imageIndex)];
                if (existingIndex != kInvalidMaterialTextureIndex) {
                    outTextureIndexMap[textureIndex] = existingIndex;
                    continue;
                }
            }
            if (materialTextures.size() >= kScenePathTraceMaxMaterialTextures) {
                log = "ScenePathTracePass exceeded the material texture descriptor limit";
                return makeError(Error::Unsupported);
            }

            DecodedMaterialTexture decodedTexture;
            const auto decodeBegin = SceneResourceLogClock::now();
            if (!decodeSceneTexture(loadedScene, textureIndex, decodedTexture, log)) {
                spdlog::info(
                    "[SceneResources] Material texture {} skipped during decode in {:.2f} ms",
                    textureIndex,
                    sceneResourceElapsedMilliseconds(decodeBegin));
                continue;
            }
            if (decodedTexture.pixels.empty() && decodedTexture.preparedMips == nullptr) {
                spdlog::info(
                    "[SceneResources] Material texture {} decoded empty payload in {:.2f} ms",
                    textureIndex,
                    sceneResourceElapsedMilliseconds(decodeBegin));
                continue;
            }
            spdlog::info(
                "[SceneResources] Material texture {} decoded '{}' {}x{} bytes={} in {:.2f} ms",
                textureIndex,
                decodedTexture.label,
                decodedTexture.width,
                decodedTexture.height,
                decodedTexture.pixels.size(),
                sceneResourceElapsedMilliseconds(decodeBegin));
            ++decodedTextureCount;
            if (decodedTexture.preparedMips != nullptr) {
                for (const scene::RenderImage::Mip& mip : *decodedTexture.preparedMips) {
                    decodedTextureBytes += mip.pixels.size();
                }
            } else {
                decodedTextureBytes += decodedTexture.pixels.size();
            }

            ScenePathTraceMaterialTexture materialTexture;
            const auto createBegin = SceneResourceLogClock::now();
            result = createMaterialTexture(
                device,
                stagingArena,
                decodedTexture.pixels.data(),
                decodedTexture.width,
                decodedTexture.height,
                decodedTexture.label,
                materialTexture,
                log,
                decodedTexture.preparedMips);
            if (!result) {
                return result;
            }
            spdlog::info(
                "[SceneResources] Material texture {} GPU resources mipCount={} uploadBytes={} in {:.2f} ms",
                textureIndex,
                materialTexture.mipCount,
                materialTexture.byteSize,
                sceneResourceElapsedMilliseconds(createBegin));

            const uint32_t materialTextureIndex = static_cast<uint32_t>(materialTextures.size());
            outTextureIndexMap[textureIndex] = materialTextureIndex;
            if (logicalTexture.imageIndex >= 0 &&
                static_cast<size_t>(logicalTexture.imageIndex) < imageTextureIndexMap.size()) {
                imageTextureIndexMap[static_cast<size_t>(logicalTexture.imageIndex)] = materialTextureIndex;
            }
            materialTextures.push_back(std::move(materialTexture));
        }

        TextureView* fallbackView = materialTextures.front().view.get();
        if (fallbackView == nullptr) {
            return makeError(Error::Failure);
        }
        materialTextureViews.fill(fallbackView);
        for (uint32_t textureIndex = 0; textureIndex < materialTextures.size(); ++textureIndex) {
            if (materialTextures[textureIndex].view == nullptr) {
                return makeError(Error::Failure);
            }
            materialTextureViews[textureIndex] = materialTextures[textureIndex].view.get();
        }
        materialTextureCount = static_cast<uint32_t>(materialTextures.size());
        spdlog::info(
            "[SceneResources] Material textures prepared decoded={} decodedBytes={} descriptorCount={}",
            decodedTextureCount,
            decodedTextureBytes,
            materialTextureCount);
        return {};
    }

    Result beginMaterialTextureBuild(
        Device& device,
        const scene::Scene& loadedScene,
        std::string& log)
    {
        Result result = neuralTextures.prepare(device, loadedScene, log);
        if (!result) {
            return result;
        }
        materialTextures.clear();
        materialTextureViews.fill(nullptr);
        materialTextureCount = 0;
        textureIndexMap.assign(loadedScene.textures().size(), kInvalidMaterialTextureIndex);
        asyncImageTextureIndexMap.assign(
            loadedScene.images().size(),
            kInvalidMaterialTextureIndex);
        asyncReferencedTextures = referencedMaterialTextures(loadedScene);
        asyncTextureCursor = 0;

        const uint8_t fallbackPixels[4] = {255, 255, 255, 255};
        ScenePathTraceMaterialTexture fallbackTexture;
        result = createMaterialTexture(
            device,
            stagingArena,
            fallbackPixels,
            1,
            1,
            "fallback",
            fallbackTexture,
            log);
        if (!result) {
            return result;
        }
        materialTextures.push_back(std::move(fallbackTexture));
        return {};
    }

    Result buildMaterialTextureStep(
        Device& device,
        const scene::Scene& loadedScene,
        bool& complete,
        std::string& log)
    {
        complete = false;
        if (asyncTextureCursor < loadedScene.textures().size()) {
            const uint32_t textureIndex = static_cast<uint32_t>(asyncTextureCursor++);
            const scene::RenderTexture& logicalTexture = loadedScene.textures()[textureIndex];
            if (textureIndex >= asyncReferencedTextures.size() ||
                !asyncReferencedTextures[textureIndex]) {
                return {};
            }
            if (neuralTextures.logicalTextureSetIndex(textureIndex) !=
                kInvalidNeuralTextureSetIndex) {
                return {};
            }
            if (logicalTexture.imageIndex >= 0 &&
                static_cast<size_t>(logicalTexture.imageIndex) < asyncImageTextureIndexMap.size()) {
                const uint32_t existingIndex =
                    asyncImageTextureIndexMap[static_cast<size_t>(logicalTexture.imageIndex)];
                if (existingIndex != kInvalidMaterialTextureIndex) {
                    textureIndexMap[textureIndex] = existingIndex;
                    return {};
                }
            }
            if (materialTextures.size() >= kScenePathTraceMaxMaterialTextures) {
                log = "ScenePathTracePass exceeded the material texture descriptor limit";
                return makeError(Error::Unsupported);
            }

            DecodedMaterialTexture decodedTexture;
            if (!decodeSceneTexture(loadedScene, textureIndex, decodedTexture, log) ||
                (decodedTexture.pixels.empty() && decodedTexture.preparedMips == nullptr)) {
                return {};
            }

            ScenePathTraceMaterialTexture materialTexture;
            Result result = createMaterialTexture(
                device,
                stagingArena,
                decodedTexture.pixels.data(),
                decodedTexture.width,
                decodedTexture.height,
                decodedTexture.label,
                materialTexture,
                log,
                decodedTexture.preparedMips);
            if (!result) {
                return result;
            }
            const uint32_t materialTextureIndex = static_cast<uint32_t>(materialTextures.size());
            textureIndexMap[textureIndex] = materialTextureIndex;
            if (logicalTexture.imageIndex >= 0 &&
                static_cast<size_t>(logicalTexture.imageIndex) < asyncImageTextureIndexMap.size()) {
                asyncImageTextureIndexMap[static_cast<size_t>(logicalTexture.imageIndex)] =
                    materialTextureIndex;
            }
            materialTextures.push_back(std::move(materialTexture));
            return {};
        }

        TextureView* fallbackView = materialTextures.front().view.get();
        if (fallbackView == nullptr) {
            return makeError(Error::Failure);
        }
        materialTextureViews.fill(fallbackView);
        for (uint32_t textureIndex = 0; textureIndex < materialTextures.size(); ++textureIndex) {
            if (materialTextures[textureIndex].view == nullptr) {
                return makeError(Error::Failure);
            }
            materialTextureViews[textureIndex] = materialTextures[textureIndex].view.get();
        }
        materialTextureCount = static_cast<uint32_t>(materialTextures.size());
        asyncImageTextureIndexMap.clear();
        complete = true;
        return {};
    }

    Result uploadTexture(CommandBuffer& commandBuffer, ScenePathTraceMaterialTexture& texture)
    {
        if (texture.uploaded) {
            return {};
        }
        if (texture.mipUploads.empty() || texture.texture == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        TextureBarrierDesc toTransfer{
            .texture = texture.texture.get(),
            .before = texture.state,
            .after = ResourceState::TransferDestination,
            .baseMip = 0,
            .mipCount = texture.mipCount,
            .baseLayer = 0,
            .layerCount = 1,
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = &toTransfer,
            .textureCount = 1,
        });
        texture.state = ResourceState::TransferDestination;

        for (uint32_t mipIndex = 0; mipIndex < texture.mipUploads.size(); ++mipIndex) {
            const ScenePathTraceTextureMipUpload& upload = texture.mipUploads[mipIndex];
            if (texture.uploadBuffer == nullptr || upload.width == 0 || upload.height == 0) {
                return makeError(Error::InvalidArgument);
            }

            commandBuffer.copyBufferToTexture(BufferTextureCopyDesc{
                .buffer = texture.uploadBuffer.get(),
                .texture = texture.texture.get(),
                .bufferOffset = texture.uploadBufferOffset + upload.bufferOffset,
                .width = upload.width,
                .height = upload.height,
                .depth = 1,
                .mipLevel = mipIndex,
                .baseLayer = 0,
            });
        }

        texture.uploaded = true;
        return {};
    }

    void transitionUploadsForRendering(CommandBuffer& commandBuffer)
    {
        auto transitionTexture = [&commandBuffer](ScenePathTraceMaterialTexture& texture) {
            if (texture.texture == nullptr || texture.state != ResourceState::TransferDestination) {
                return;
            }
            TextureBarrierDesc toShaderRead{
                .texture = texture.texture.get(),
                .before = ResourceState::TransferDestination,
                .after = ResourceState::ShaderRead,
                .baseMip = 0,
                .mipCount = texture.mipCount,
                .baseLayer = 0,
                .layerCount = 1,
            };
            commandBuffer.barrier(BarrierDesc{
                .textures = &toShaderRead,
                .textureCount = 1,
            });
            texture.state = ResourceState::ShaderRead;
        };

        for (ScenePathTraceMaterialTexture& texture : materialTextures) {
            transitionTexture(texture);
        }

        std::vector<BufferBarrierDesc> bufferBarriers;
        bufferBarriers.reserve(bufferUploads.size());
        for (const ScenePathTraceBufferUpload& upload : bufferUploads) {
            bufferBarriers.push_back(BufferBarrierDesc{
                .buffer = upload.destination,
                .before = ResourceState::TransferDestination,
                .after = ResourceState::General,
                .offset = 0,
                .size = upload.byteSize,
            });
        }
        if (!bufferBarriers.empty()) {
            commandBuffer.barrier(BarrierDesc{
                .buffers = bufferBarriers.data(),
                .bufferCount = static_cast<uint32_t>(bufferBarriers.size()),
            });
        }
    }

    Result uploadMaterialTextures(CommandBuffer& commandBuffer)
    {
        if (textureUploadsSubmitted) {
            (void)commandBuffer;
            retireCompletedTextureUploads();
            return {};
        }
        if (materialTextures.empty()) {
            return makeError(Error::InvalidArgument);
        }

        for (ScenePathTraceMaterialTexture& texture : materialTextures) {
            Result result = uploadTexture(commandBuffer, texture);
            if (!result) {
                return result;
            }
        }
        return {};
    }

    void resetGpuBuffers()
    {
        waitForTextureUploads();
        vertexBuffer.reset();
        indexBuffer.reset();
        primitiveBuffer.reset();
        instanceBuffer.reset();
        materialBuffer.reset();
        neuralTextures.clear();
        materialTextures.clear();
        materialTextureViews.fill(nullptr);
        materialTextureCount = 0;
    }

    void clear()
    {
        resetGpuBuffers();
        rtxBuilder.clear();
        drawBounds = scene::Bounds{};
        scenePath.clear();
        prepared = false;
        asyncPrepareStage = AsyncPrepareStage::Idle;
        partialUploadResumeStage = AsyncPrepareStage::Idle;
        asyncScene = nullptr;
        asyncScenePath.clear();
        sourceResourceIdentity = 0;
        sourceStructuralRevision = 0;
        sourceTransformRevision = 0;
        sourceVisibilityRevision = 0;
        asyncSourceResourceIdentity = 0;
        asyncSourceStructuralRevision = 0;
        asyncSourceTransformRevision = 0;
        asyncSourceVisibilityRevision = 0;
        asyncGpuScene = ScenePathTraceGpuScene{};
        asyncReferencedTextures.clear();
        asyncBufferStep = 0;
    }

    bool valid() const
    {
        return prepared &&
            rtxBuilder.valid() &&
            drawBounds.valid &&
            vertexBuffer != nullptr &&
            indexBuffer != nullptr &&
            primitiveBuffer != nullptr &&
            instanceBuffer != nullptr &&
            materialBuffer != nullptr &&
            !materialTextures.empty() &&
            materialTextureViews[0] != nullptr;
    }

    bool sourceTopologyMatches(const scene::Scene& sourceScene) const
    {
        return sourceResourceIdentity == sourceScene.resourceIdentity() &&
            sourceStructuralRevision ==
                sourceScene.sceneGraph().structuralRevision() &&
            sourceVisibilityRevision == sourceScene.visibilityRevision();
    }

    void stampSource(const scene::Scene& sourceScene)
    {
        sourceResourceIdentity = sourceScene.resourceIdentity();
        sourceStructuralRevision =
            sourceScene.sceneGraph().structuralRevision();
        sourceTransformRevision = sourceScene.transformRevision();
        sourceVisibilityRevision = sourceScene.visibilityRevision();
    }

    SceneAccelerationStructureBuilder rtxBuilder;
    Device* device = nullptr;
    Queue* graphicsQueue = nullptr;
    scene::Bounds drawBounds;
    std::filesystem::path scenePath;
    bool prepared = false;
    uint64_t revision = 0;
    uint64_t sourceResourceIdentity = 0;
    uint64_t sourceStructuralRevision = 0;
    uint64_t sourceTransformRevision = 0;
    uint64_t sourceVisibilityRevision = 0;
    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    std::unique_ptr<Buffer> primitiveBuffer;
    std::unique_ptr<Buffer> instanceBuffer;
    std::unique_ptr<Buffer> materialBuffer;
    NeuralTextureResources neuralTextures;
    SceneUploadStagingArena stagingArena;
    std::vector<ScenePathTraceBufferUpload> bufferUploads;
    std::vector<ScenePathTraceMaterialTexture> materialTextures;
    std::vector<uint32_t> textureIndexMap;
    std::vector<uint32_t> asyncImageTextureIndexMap;
    std::vector<bool> asyncReferencedTextures;
    size_t asyncTextureCursor = 0;
    std::array<TextureView*, kScenePathTraceMaxMaterialTextures> materialTextureViews{};
    uint32_t materialTextureCount = 0;
    std::unique_ptr<CommandPool> textureUploadCommandPool;
    std::unique_ptr<CommandBuffer> textureUploadCommandBuffer;
    std::unique_ptr<CommandPool> textureAcquireCommandPool;
    std::unique_ptr<CommandBuffer> textureAcquireCommandBuffer;
    std::unique_ptr<Semaphore> textureUploadTimeline;
    uint64_t textureUploadTimelineValue = 1;
    bool textureUploadsSubmitted = false;
    AsyncPrepareStage asyncPrepareStage = AsyncPrepareStage::Idle;
    AsyncPrepareStage partialUploadResumeStage = AsyncPrepareStage::Idle;
    const scene::Scene* asyncScene = nullptr;
    std::filesystem::path asyncScenePath;
    uint64_t asyncSourceResourceIdentity = 0;
    uint64_t asyncSourceStructuralRevision = 0;
    uint64_t asyncSourceTransformRevision = 0;
    uint64_t asyncSourceVisibilityRevision = 0;
    ScenePathTraceGpuScene asyncGpuScene;
    uint32_t asyncBufferStep = 0;
};
ScenePathTraceResources::ScenePathTraceResources() :
    impl_(std::make_shared<Impl>())
{
}

ScenePathTraceResources::~ScenePathTraceResources() = default;

ScenePathTraceResources::ScenePathTraceResources(ScenePathTraceResources&&) noexcept = default;

ScenePathTraceResources& ScenePathTraceResources::operator=(ScenePathTraceResources&&) noexcept = default;

Result ScenePathTraceResources::prepare(
    Device& device,
    Queue& graphicsQueue,
    const RenderGraphProperties& properties,
    const scene::Scene* runtimeScene,
    std::string& log)
{
    impl_->device = &device;
    impl_->graphicsQueue = &graphicsQueue;
    const std::filesystem::path path = scenePathFromProperties(properties);
    const scene::Scene* boundScene = runtimeSceneForPath(runtimeScene, path);
    if (impl_->valid() && impl_->scenePath == path &&
        boundScene != nullptr && impl_->sourceTopologyMatches(*boundScene) &&
        impl_->sourceTransformRevision != boundScene->transformRevision()) {
        return syncRuntimeScene(boundScene, log);
    }
    if (impl_->valid() && impl_->scenePath == path &&
        (boundScene == nullptr ||
         (impl_->sourceTopologyMatches(*boundScene) &&
          impl_->sourceTransformRevision == boundScene->transformRevision()))) {
        spdlog::info("[SceneResources] Reuse prepared scene='{}'", path.string());
        return {};
    }

    SceneResourceLogScope prepareScope("prepare scene='" + path.string() + "'");
    impl_->clear();

    scene::SceneDocument fallbackScene;
    if (boundScene == nullptr) {
        SceneResourceLogScope scope("load scene for render pass resources");
        if (!fallbackScene.load(path)) {
            log = "ScenePathTracePass failed to load scene: " +
                fallbackScene.lastLoadResult().error;
            return makeError(Error::Failure);
        }
        boundScene = &fallbackScene;
    }
    const scene::Scene& loadedScene = *boundScene;
    if (!loadedScene.bounds().valid) {
        log = "ScenePathTracePass scene bounds are unavailable";
        return makeError(Error::Failure);
    }
    const scene::SceneStats& sceneStats = loadedScene.stats();
    spdlog::info(
        "[SceneResources] Loaded scene stats nodes={} renderNodes={} primitives={} triangles={} images={} textures={}",
        loadedScene.nodes().size(),
        sceneStats.renderNodeCount,
        sceneStats.primitiveCount,
        sceneStats.triangleCount,
        sceneStats.imageCount,
        sceneStats.textureCount);

    std::vector<uint32_t> textureIndexMap;
    Result result;
    {
        SceneResourceLogScope scope("build material textures");
        result = impl_->buildMaterialTextures(device, loadedScene, textureIndexMap, log);
    }
    if (!result) {
        impl_->clear();
        return result;
    }
    impl_->textureIndexMap = textureIndexMap;
    ScenePathTraceGpuScene gpuScene;
    {
        SceneResourceLogScope scope("build GPU scene payload");
        if (!buildGpuScene(
                loadedScene,
                textureIndexMap,
                impl_->neuralTextures.logicalTextureSetIndices(),
                gpuScene,
                log)) {
            impl_->clear();
            return makeError(Error::Failure);
        }
    }
    spdlog::info(
        "[SceneResources] GPU scene payload vertices={} indices={} primitives={} instances={} materials={}",
        gpuScene.vertices.size(),
        gpuScene.indices.size(),
        gpuScene.primitives.size(),
        gpuScene.instances.size(),
        gpuScene.materials.size());

    std::string rtxLog;
    {
        SceneResourceLogScope scope("build ray tracing acceleration structures for render pass");
        Queue* accelerationQueue = device.getQueue(QueueType::Compute);
        if (accelerationQueue == nullptr) {
            accelerationQueue = &graphicsQueue;
        }
        result = impl_->rtxBuilder.beginBuild(device, *accelerationQueue, loadedScene, rtxLog);
    }
    if (!result) {
        appendLogBlock(log, rtxLog);
        impl_->clear();
        return result;
    }
    appendLogBlock(log, rtxLog);

    {
        SceneResourceLogScope scope("upload GPU scene storage buffers");
        result = uploadStorageBuffer(
            device,
            gpuScene.vertices.data(),
            static_cast<uint64_t>(gpuScene.vertices.size() * sizeof(ScenePathTraceGpuVertex)),
            sizeof(ScenePathTraceGpuVertex),
            impl_->vertexBuffer,
            log,
            "ScenePathTracePass vertices",
            &impl_->bufferUploads,
            &impl_->stagingArena);
        if (!result) {
            impl_->clear();
            return result;
        }
        result = uploadStorageBuffer(
            device,
            gpuScene.indices.data(),
            static_cast<uint64_t>(gpuScene.indices.size() * sizeof(uint32_t)),
            sizeof(uint32_t),
            impl_->indexBuffer,
            log,
            "ScenePathTracePass indices",
            &impl_->bufferUploads,
            &impl_->stagingArena);
        if (!result) {
            impl_->clear();
            return result;
        }
        result = uploadStorageBuffer(
            device,
            gpuScene.primitives.data(),
            static_cast<uint64_t>(gpuScene.primitives.size() * sizeof(ScenePathTraceGpuPrimitive)),
            sizeof(ScenePathTraceGpuPrimitive),
            impl_->primitiveBuffer,
            log,
            "ScenePathTracePass primitives",
            &impl_->bufferUploads,
            &impl_->stagingArena);
        if (!result) {
            impl_->clear();
            return result;
        }
        result = uploadStorageBuffer(
            device,
            gpuScene.instances.data(),
            static_cast<uint64_t>(gpuScene.instances.size() * sizeof(ScenePathTraceGpuInstance)),
            sizeof(ScenePathTraceGpuInstance),
            impl_->instanceBuffer,
            log,
            "ScenePathTracePass instances",
            &impl_->bufferUploads,
            &impl_->stagingArena);
        if (!result) {
            impl_->clear();
            return result;
        }
        result = uploadStorageBuffer(
            device,
            gpuScene.materials.data(),
            static_cast<uint64_t>(gpuScene.materials.size() * sizeof(ScenePathTraceGpuMaterial)),
            sizeof(ScenePathTraceGpuMaterial),
            impl_->materialBuffer,
            log,
            "ScenePathTracePass materials",
            &impl_->bufferUploads,
            &impl_->stagingArena);
        if (!result) {
            impl_->clear();
            return result;
        }
    }

    {
        SceneResourceLogScope scope("submit asynchronous scene uploads");
        result = impl_->submitTextureUploads(device, graphicsQueue, log);
    }
    if (!result) {
        impl_->clear();
        return result;
    }

    impl_->drawBounds = loadedScene.bounds();
    impl_->scenePath = path;
    impl_->stampSource(loadedScene);
    impl_->prepared = true;
    ++impl_->revision;
    spdlog::info(
        "[SceneResources] Prepared scene resources revision={} materialTextures={}",
        impl_->revision,
        impl_->materialTextureCount);
    return {};
}

Result ScenePathTraceResources::beginPrepareAsync(
    Device& device,
    Queue& graphicsQueue,
    const RenderGraphProperties& properties,
    const scene::Scene& runtimeScene,
    std::string& log)
{
    const std::filesystem::path path = scenePathFromProperties(properties);
    const scene::Scene* boundScene = runtimeSceneForPath(&runtimeScene, path);
    if (boundScene == nullptr || !boundScene->bounds().valid) {
        log = "Asynchronous scene preparation requires a matching valid runtime scene";
        return makeError(Error::InvalidArgument);
    }
    if (impl_->valid() && impl_->scenePath == path &&
        impl_->sourceTopologyMatches(*boundScene)) {
        return impl_->sourceTransformRevision == boundScene->transformRevision()
            ? Result{}
            : syncRuntimeScene(boundScene, log);
    }

    impl_->clear();
    impl_->device = &device;
    impl_->graphicsQueue = &graphicsQueue;
    impl_->asyncScene = boundScene;
    impl_->asyncScenePath = path;
    impl_->asyncSourceResourceIdentity = boundScene->resourceIdentity();
    impl_->asyncSourceStructuralRevision =
        boundScene->sceneGraph().structuralRevision();
    impl_->asyncSourceTransformRevision = boundScene->transformRevision();
    impl_->asyncSourceVisibilityRevision = boundScene->visibilityRevision();
    Result result = impl_->beginMaterialTextureBuild(device, *boundScene, log);
    if (!result) {
        impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
        return result;
    }
    impl_->asyncPrepareStage = Impl::AsyncPrepareStage::MaterialTextures;
    return {};
}

Result ScenePathTraceResources::pumpPrepareAsync(
    double budgetMilliseconds,
    bool& complete,
    scene::SceneLoadProgress& progress,
    std::string& log)
{
    complete = false;
    progress.status = scene::SceneLoadStatus::Running;
    if (impl_->asyncPrepareStage == Impl::AsyncPrepareStage::Ready || impl_->valid()) {
        complete = true;
        progress.status = scene::SceneLoadStatus::Succeeded;
        progress.phase = scene::SceneLoadPhase::Completed;
        progress.fraction = 1.0f;
        return {};
    }
    if (impl_->asyncPrepareStage == Impl::AsyncPrepareStage::Idle ||
        impl_->asyncPrepareStage == Impl::AsyncPrepareStage::Failed ||
        impl_->device == nullptr ||
        impl_->graphicsQueue == nullptr ||
        impl_->asyncScene == nullptr) {
        progress.status = scene::SceneLoadStatus::Failed;
        progress.phase = scene::SceneLoadPhase::Failed;
        progress.error = log.empty() ? "Scene resource preparation is not active" : log;
        return makeError(Error::InvalidArgument);
    }
    if (impl_->asyncScene->resourceIdentity() !=
            impl_->asyncSourceResourceIdentity ||
        impl_->asyncScene->sceneGraph().structuralRevision() !=
            impl_->asyncSourceStructuralRevision ||
        impl_->asyncScene->visibilityRevision() !=
            impl_->asyncSourceVisibilityRevision) {
        log = "Scene topology changed during asynchronous resource preparation.";
        impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
        progress.status = scene::SceneLoadStatus::Failed;
        progress.phase = scene::SceneLoadPhase::Failed;
        progress.error = log;
        return makeError(Error::InvalidArgument);
    }

    const auto begin = SceneResourceLogClock::now();
    const double budget = std::max(budgetMilliseconds, 0.1);
    Result result;
    while (sceneResourceElapsedMilliseconds(begin) < budget) {
        switch (impl_->asyncPrepareStage) {
        case Impl::AsyncPrepareStage::MaterialTextures: {
            const uint64_t nextUploadBytes =
                impl_->nextMaterialTextureUploadByteSize(*impl_->asyncScene);
            const uint32_t nextUploadRegions =
                impl_->nextMaterialTextureUploadRegionCount(*impl_->asyncScene);
            if (impl_->shouldFlushBefore(nextUploadBytes, nextUploadRegions)) {
                impl_->partialUploadResumeStage = Impl::AsyncPrepareStage::MaterialTextures;
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::SubmitPartialUploads;
                continue;
            }
            bool texturesComplete = false;
            result = impl_->buildMaterialTextureStep(
                *impl_->device,
                *impl_->asyncScene,
                texturesComplete,
                log);
            if (!result) {
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return result;
            }
            progress.phase = scene::SceneLoadPhase::GpuUpload;
            progress.completedUnits = impl_->asyncTextureCursor;
            progress.totalUnits = impl_->asyncScene->textures().size();
            progress.fraction = 0.65f + 0.10f * static_cast<float>(impl_->asyncTextureCursor) /
                static_cast<float>(std::max<size_t>(impl_->asyncScene->textures().size(), 1u));
            if (texturesComplete) {
                impl_->partialUploadResumeStage = Impl::AsyncPrepareStage::GpuPayload;
                impl_->asyncPrepareStage = impl_->pendingUploadByteSize() > 0
                    ? Impl::AsyncPrepareStage::SubmitPartialUploads
                    : Impl::AsyncPrepareStage::GpuPayload;
            } else if (impl_->uploadBatchLimitReached()) {
                impl_->partialUploadResumeStage = Impl::AsyncPrepareStage::MaterialTextures;
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::SubmitPartialUploads;
            }
            return {};
        }
        case Impl::AsyncPrepareStage::GpuPayload:
            if (!buildGpuScene(
                    *impl_->asyncScene,
                    impl_->textureIndexMap,
                    impl_->neuralTextures.logicalTextureSetIndices(),
                    impl_->asyncGpuScene,
                    log)) {
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return makeError(Error::Failure);
            }
            impl_->asyncPrepareStage = Impl::AsyncPrepareStage::AccelerationStructure;
            progress.phase = scene::SceneLoadPhase::GpuUpload;
            progress.fraction = 0.82f;
            return {};
        case Impl::AsyncPrepareStage::AccelerationStructure: {
            Queue* accelerationQueue = impl_->device->getQueue(QueueType::Compute);
            if (accelerationQueue == nullptr) {
                accelerationQueue = impl_->graphicsQueue;
            }
            result = impl_->rtxBuilder.beginBuild(
                *impl_->device,
                *accelerationQueue,
                *impl_->asyncScene,
                log);
            if (!result) {
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return result;
            }
            impl_->asyncBufferStep = 0;
            impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Buffers;
            progress.phase = scene::SceneLoadPhase::AccelerationStructures;
            progress.fraction = 0.87f;
            return {};
        }
        case Impl::AsyncPrepareStage::Buffers: {
            const void* data = nullptr;
            uint64_t byteSize = 0;
            uint32_t stride = 0;
            std::unique_ptr<Buffer>* destination = nullptr;
            const char* label = nullptr;
            switch (impl_->asyncBufferStep) {
            case 0:
                data = impl_->asyncGpuScene.vertices.data();
                byteSize = impl_->asyncGpuScene.vertices.size() * sizeof(ScenePathTraceGpuVertex);
                stride = sizeof(ScenePathTraceGpuVertex);
                destination = &impl_->vertexBuffer;
                label = "ScenePathTracePass vertices";
                break;
            case 1:
                data = impl_->asyncGpuScene.indices.data();
                byteSize = impl_->asyncGpuScene.indices.size() * sizeof(uint32_t);
                stride = sizeof(uint32_t);
                destination = &impl_->indexBuffer;
                label = "ScenePathTracePass indices";
                break;
            case 2:
                data = impl_->asyncGpuScene.primitives.data();
                byteSize = impl_->asyncGpuScene.primitives.size() * sizeof(ScenePathTraceGpuPrimitive);
                stride = sizeof(ScenePathTraceGpuPrimitive);
                destination = &impl_->primitiveBuffer;
                label = "ScenePathTracePass primitives";
                break;
            case 3:
                data = impl_->asyncGpuScene.instances.data();
                byteSize = impl_->asyncGpuScene.instances.size() * sizeof(ScenePathTraceGpuInstance);
                stride = sizeof(ScenePathTraceGpuInstance);
                destination = &impl_->instanceBuffer;
                label = "ScenePathTracePass instances";
                break;
            case 4:
                data = impl_->asyncGpuScene.materials.data();
                byteSize = impl_->asyncGpuScene.materials.size() * sizeof(ScenePathTraceGpuMaterial);
                stride = sizeof(ScenePathTraceGpuMaterial);
                destination = &impl_->materialBuffer;
                label = "ScenePathTracePass materials";
                break;
            default:
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::SubmitUploads;
                continue;
            }
            if (impl_->shouldFlushBefore(byteSize, 1)) {
                impl_->partialUploadResumeStage = Impl::AsyncPrepareStage::Buffers;
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::SubmitPartialUploads;
                continue;
            }
            result = uploadStorageBuffer(
                *impl_->device,
                data,
                byteSize,
                stride,
                *destination,
                log,
                label,
                &impl_->bufferUploads,
                &impl_->stagingArena);
            if (!result) {
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return result;
            }
            ++impl_->asyncBufferStep;
            progress.phase = scene::SceneLoadPhase::GpuUpload;
            progress.fraction = 0.88f + 0.03f * static_cast<float>(impl_->asyncBufferStep) / 5.0f;
            if (impl_->uploadBatchLimitReached()) {
                impl_->partialUploadResumeStage = Impl::AsyncPrepareStage::Buffers;
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::SubmitPartialUploads;
            }
            return {};
        }
        case Impl::AsyncPrepareStage::SubmitPartialUploads:
            result = impl_->submitTextureUploads(
                *impl_->device,
                *impl_->graphicsQueue,
                log);
            if (!result) {
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return result;
            }
            impl_->asyncPrepareStage = Impl::AsyncPrepareStage::WaitForPartialUploads;
            progress.phase = scene::SceneLoadPhase::GpuUpload;
            progress.fraction = std::max(progress.fraction, 0.70f);
            return {};
        case Impl::AsyncPrepareStage::WaitForPartialUploads:
            progress.phase = scene::SceneLoadPhase::GpuUpload;
            progress.fraction = std::max(progress.fraction, 0.70f);
            if (!impl_->textureUploadsReady()) {
                return {};
            }
            impl_->retireCompletedTextureUploads();
            impl_->asyncPrepareStage = impl_->partialUploadResumeStage;
            impl_->partialUploadResumeStage = Impl::AsyncPrepareStage::Idle;
            return {};
        case Impl::AsyncPrepareStage::SubmitUploads:
            result = impl_->submitTextureUploads(
                *impl_->device,
                *impl_->graphicsQueue,
                log);
            if (!result) {
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return result;
            }
            impl_->asyncPrepareStage = Impl::AsyncPrepareStage::WaitForGpu;
            progress.phase = scene::SceneLoadPhase::AccelerationStructures;
            progress.fraction = 0.94f;
            return {};
        case Impl::AsyncPrepareStage::WaitForGpu:
            progress.phase = scene::SceneLoadPhase::AccelerationStructures;
            progress.fraction = 0.97f;
            {
                bool accelerationStructuresComplete = false;
                std::string rtxLog;
                result = impl_->rtxBuilder.pollBuild(
                    accelerationStructuresComplete,
                    rtxLog);
                appendLogBlock(log, rtxLog);
                if (!result) {
                    impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                    progress.status = scene::SceneLoadStatus::Failed;
                    progress.phase = scene::SceneLoadPhase::Failed;
                    progress.error = rtxLog.empty()
                        ? "Scene acceleration-structure build failed."
                        : rtxLog;
                    return result;
                }
                if (!impl_->textureUploadsReady() || !accelerationStructuresComplete) {
                    return {};
                }
            }
            impl_->retireCompletedTextureUploads();
            impl_->drawBounds = impl_->asyncScene->bounds();
            impl_->scenePath = impl_->asyncScenePath;
            impl_->sourceResourceIdentity = impl_->asyncSourceResourceIdentity;
            impl_->sourceStructuralRevision = impl_->asyncSourceStructuralRevision;
            impl_->sourceTransformRevision = impl_->asyncSourceTransformRevision;
            impl_->sourceVisibilityRevision = impl_->asyncSourceVisibilityRevision;
            impl_->prepared = true;
            ++impl_->revision;
            impl_->asyncScene = nullptr;
            impl_->asyncGpuScene = ScenePathTraceGpuScene{};
            impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Ready;
            complete = true;
            progress.status = scene::SceneLoadStatus::Succeeded;
            progress.phase = scene::SceneLoadPhase::Finalizing;
            progress.fraction = 0.97f;
            return {};
        case Impl::AsyncPrepareStage::Ready:
        case Impl::AsyncPrepareStage::Idle:
        case Impl::AsyncPrepareStage::Failed:
            break;
        }
        break;
    }
    return {};
}

bool ScenePathTraceResources::preparing() const
{
    return impl_ != nullptr &&
        impl_->asyncPrepareStage != Impl::AsyncPrepareStage::Idle &&
        impl_->asyncPrepareStage != Impl::AsyncPrepareStage::Ready &&
        impl_->asyncPrepareStage != Impl::AsyncPrepareStage::Failed;
}

Result ScenePathTraceResources::syncRuntimeScene(
    const scene::Scene* runtimeScene,
    std::string& log)
{
    log.clear();
    const scene::Scene* boundScene = runtimeSceneForPath(runtimeScene, impl_->scenePath);
    if (boundScene == nullptr || !impl_->valid()) {
        return {};
    }
    if (impl_->device == nullptr || impl_->graphicsQueue == nullptr) {
        log = "Scene resources have no device or graphics queue for a runtime transform update.";
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->sourceTopologyMatches(*boundScene)) {
        Device& device = *impl_->device;
        Queue& graphicsQueue = *impl_->graphicsQueue;
        const RenderGraphProperties properties{
            {"path", impl_->scenePath.string()},
        };
        Result result = prepare(
            device,
            graphicsQueue,
            properties,
            boundScene,
            log);
        if (!result) {
            return result;
        }

        // prepare() deliberately submits its acceleration-structure build
        // asynchronously. A direct runtime sync, however, has no async pump
        // to advance that build, so complete it before reporting success.
        Queue* accelerationQueue = device.getQueue(QueueType::Compute);
        if (accelerationQueue == nullptr) {
            accelerationQueue = &graphicsQueue;
        }
        while (impl_->rtxBuilder.buildState() ==
            SceneAccelerationStructureBuildState::Building) {
            result = accelerationQueue->waitIdle();
            if (!result) {
                appendLogBlock(
                    log,
                    resultMessage(
                        "Queue::waitIdle(runtime scene topology rebuild)",
                        result));
                impl_->clear();
                return result;
            }
            bool accelerationStructuresComplete = false;
            std::string rtxLog;
            result = impl_->rtxBuilder.pollBuild(
                accelerationStructuresComplete,
                rtxLog);
            appendLogBlock(log, rtxLog);
            if (!result) {
                impl_->clear();
                return result;
            }
        }
        if (!impl_->valid()) {
            appendLogBlock(
                log,
                "Runtime scene topology rebuild did not produce valid scene resources.");
            impl_->clear();
            return makeError(Error::Failure);
        }
        return {};
    }
    if (impl_->sourceTransformRevision == boundScene->transformRevision()) {
        return {};
    }

    ScenePathTraceGpuScene gpuScene;
    if (!buildGpuScene(
            *boundScene,
            impl_->textureIndexMap,
            impl_->neuralTextures.logicalTextureSetIndices(),
            gpuScene,
            log)) {
        return makeError(Error::Failure);
    }
    std::string rtxLog;
    Queue* accelerationQueue = impl_->device->getQueue(QueueType::Compute);
    if (accelerationQueue == nullptr) {
        accelerationQueue = impl_->graphicsQueue;
    }
    Result result = impl_->rtxBuilder.updateInstanceTransforms(
        *impl_->device,
        *accelerationQueue,
        *boundScene,
        rtxLog);
    appendLogBlock(log, rtxLog);
    if (!result) {
        return result;
    }
    result = uploadStorageBuffer(
        *impl_->device,
        gpuScene.instances.data(),
        static_cast<uint64_t>(gpuScene.instances.size() * sizeof(ScenePathTraceGpuInstance)),
        sizeof(ScenePathTraceGpuInstance),
        impl_->instanceBuffer,
        log,
        "ScenePathTracePass updated instances");
    if (!result) {
        return result;
    }
    impl_->drawBounds = boundScene->bounds();
    impl_->stampSource(*boundScene);
    ++impl_->revision;
    spdlog::info(
        "[SceneResources] Updated instance transforms and refit TLAS revision={}",
        impl_->revision);
    return {};
}

Result ScenePathTraceResources::uploadMaterialTextures(CommandBuffer& commandBuffer)
{
    return impl_->uploadMaterialTextures(commandBuffer);
}

void ScenePathTraceResources::clear()
{
    impl_->clear();
}

bool ScenePathTraceResources::valid() const
{
    return impl_ != nullptr && impl_->valid();
}

uint64_t ScenePathTraceResources::revision() const
{
    return impl_->revision;
}

const scene::Bounds& ScenePathTraceResources::bounds() const
{
    return impl_->drawBounds;
}

SceneAccelerationStructureBuilder& ScenePathTraceResources::accelerationStructure()
{
    return impl_->rtxBuilder;
}

const SceneAccelerationStructureBuilder& ScenePathTraceResources::accelerationStructure() const
{
    return impl_->rtxBuilder;
}

Buffer* ScenePathTraceResources::vertexBuffer() const
{
    return impl_->vertexBuffer.get();
}

Buffer* ScenePathTraceResources::indexBuffer() const
{
    return impl_->indexBuffer.get();
}

Buffer* ScenePathTraceResources::primitiveBuffer() const
{
    return impl_->primitiveBuffer.get();
}

Buffer* ScenePathTraceResources::instanceBuffer() const
{
    return impl_->instanceBuffer.get();
}

Buffer* ScenePathTraceResources::materialBuffer() const
{
    return impl_->materialBuffer.get();
}

const std::array<TextureView*, kScenePathTraceMaxMaterialTextures>& ScenePathTraceResources::materialTextureViews() const
{
    return impl_->materialTextureViews;
}

uint32_t ScenePathTraceResources::materialTextureCount() const
{
    return impl_->materialTextureCount;
}

const NeuralTextureResources& ScenePathTraceResources::neuralTextures() const
{
    return impl_->neuralTextures;
}

bool ScenePathTraceResources::textureUploadsReady() const
{
    return impl_->textureUploadsReady();
}

bool ScenePathTraceResources::gpuWorkComplete()
{
    bool accelerationStructureComplete =
        impl_->rtxBuilder.buildState() != SceneAccelerationStructureBuildState::Building;
    Result result;
    if (!accelerationStructureComplete) {
        std::string log;
        result = impl_->rtxBuilder.pollBuild(accelerationStructureComplete, log);
        if (!result && !log.empty()) {
            spdlog::error("[SceneResources] {}", log);
        }
    }
    return impl_->textureUploadsReady() && (accelerationStructureComplete || !result);
}

} // namespace metallic::render
