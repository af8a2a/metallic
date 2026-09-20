#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Render/Streamer/Ktx2Texture.h"
#include "Runtime/Render/RenderFrameContext.h"
#include <optional>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
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

struct UploadCpuTimer {
    double& elapsed;
    SceneResourceLogClock::time_point begin = SceneResourceLogClock::now();
    ~UploadCpuTimer() { elapsed += sceneResourceElapsedMilliseconds(begin); }
};

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
    float specular[4] = {1.0f, 1.0f, 1.0f, 1.0f}; // RGB color, scalar weight
    TextureInfo specularTexture;
    TextureInfo specularColorTexture;
};
static_assert(sizeof(ScenePathTraceGpuMaterial) == 720);
static_assert(offsetof(ScenePathTraceGpuMaterial, specular) == 608);

struct ScenePathTraceGpuScene {
    std::vector<SceneShadingVertex> vertices;
    std::vector<std::array<float, 3>> positions;
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
    std::shared_ptr<Texture> texture;
    std::shared_ptr<TextureView> view;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t mipCount = 1;
    uint64_t byteSize = 4;
    Format format = Format::Rgba8Unorm;
    ResourceState state = ResourceState::Undefined;
    bool uploaded = false;
};

Result createSharedTexture(Device& device, const TextureDesc& desc, std::shared_ptr<Texture>& output)
{
    std::unique_ptr<Texture> texture;
    auto result = device.createTexture(desc, texture);
    output = std::move(texture);
    return result;
}
Result createSharedTextureView(Device& device, Texture& texture, const TextureViewDesc& desc, std::shared_ptr<TextureView>& output)
{
    std::unique_ptr<TextureView> view;
    auto result = device.createTextureView(texture, desc, view);
    output = std::move(view);
    return result;
}

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

    SceneUploadStagingArena() = default;
    SceneUploadStagingArena(SceneUploadStagingArena&&) noexcept = default;
    SceneUploadStagingArena& operator=(SceneUploadStagingArena&&) noexcept = default;

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
        void* mapped = nullptr;
        Result result = allocate(device, byteSize, alignment, outBuffer, outOffset, mapped, log, label);
        if (!result) {
            return result;
        }
        std::memcpy(mapped, data, static_cast<size_t>(byteSize));
        outBuffer->flush(outOffset, byteSize);
        return {};
    }

    Result allocate(
        Device& device,
        uint64_t byteSize,
        uint64_t alignment,
        std::shared_ptr<Buffer>& outBuffer,
        uint64_t& outOffset,
        void*& outMapped,
        std::string& log,
        std::string_view label)
    {
        if (byteSize == 0 || byteSize > std::numeric_limits<size_t>::max()) {
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

        selectedPage->cursor = selectedOffset + byteSize;
        outBuffer = selectedPage->buffer;
        outOffset = selectedOffset;
        outMapped = static_cast<uint8_t*>(selectedPage->mapped) + selectedOffset;
        return {};
    }

    SceneUploadStagingArena takeUsedPages()
    {
        SceneUploadStagingArena used;
        for (auto& page : pages_) {
            if (page->cursor != 0) {
                used.pages_.push_back(std::move(page));
            }
        }
        std::erase(pages_, nullptr);
        return used;
    }

    void recycle(SceneUploadStagingArena&& completed)
    {
        for (auto& page : completed.pages_) {
            page->cursor = 0;
            pages_.push_back(std::move(page));
        }
        completed.pages_.clear();
    }

    uint64_t allocatedBytes() const
    {
        uint64_t bytes = 0;
        for (const auto& page : pages_) { bytes += page->capacity; }
        return bytes;
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
    uint64_t allocationSize = 0;
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
        upload.bufferOffset = (allocationSize + uploadAlignment - 1u) /
            uploadAlignment * uploadAlignment;
        if (mipPixels.size() < upload.byteSize ||
            upload.bufferOffset > std::numeric_limits<size_t>::max() - upload.byteSize) {
            return makeError(Error::InvalidArgument);
        }
        allocationSize = upload.bufferOffset + upload.byteSize;
        outTexture.byteSize += upload.byteSize;
        outTexture.mipUploads.push_back(std::move(upload));
    }

    std::string uploadLabel = "ScenePathTracePass texture upload ";
    uploadLabel += std::string(label);
    void* mapped = nullptr;
    Result result = stagingArena.allocate(
        device,
        allocationSize,
        uploadAlignment,
        outTexture.uploadBuffer,
        outTexture.uploadBufferOffset,
        mapped,
        log,
        uploadLabel);
    if (!result) {
        return result;
    }
    // Copy each existing mip directly to its final staging offset. There is no
    // packed CPU vector to initialize, grow, or copy a second time.
    for (size_t mipIndex = 0; mipIndex < mipCount; ++mipIndex) {
        const auto& mipPixels = preparedMips != nullptr && !preparedMips->empty()
            ? (*preparedMips)[mipIndex].pixels : generatedMipChain[mipIndex].pixels;
        const auto& upload = outTexture.mipUploads[mipIndex];
        std::memcpy(static_cast<uint8_t*>(mapped) + upload.bufferOffset,
            mipPixels.data(), static_cast<size_t>(upload.byteSize));
    }
    outTexture.uploadBuffer->flush(outTexture.uploadBufferOffset, allocationSize);
    outTexture.uploadAllocationSize = allocationSize;

    result = createSharedTexture(device,
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
            .memoryDomain = MemoryBudgetDomain::MaterialTextures,
        },
        outTexture.texture);
    if (!result || outTexture.texture == nullptr) {
        log += resultMessage(std::string("createTexture(ScenePathTracePass material texture ") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }

    result = createSharedTextureView(device,
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

Result createKtxMaterialTexture(Device& device, SceneUploadStagingArena& arena,
    Ktx2MipReader& reader, std::vector<uint8_t>& bytes, SceneTextureLoadTiming& timing,
    const Ktx2TextureInfo& info, uint32_t firstMip, ScenePathTraceMaterialTexture& texture, std::string& log,
    const Ktx2PrefetchResult* prefetched = nullptr)
{
    UploadCpuTimer totalTimer{timing.totalMs};
    timing.path = info.path.string();
    texture = {};
    const auto desc = info.textureDesc(firstMip);
    texture.width = desc.width; texture.height = desc.height; texture.mipCount = desc.mipCount;
    texture.format = desc.format; texture.byteSize = info.tailBytes(firstMip);
    const uint64_t alignment = std::max<uint64_t>(16, device.capabilities().textureUploadBufferOffsetAlignment);
    uint64_t size = 0;
    for (uint32_t i = 0; i < desc.mipCount; ++i) {
        const uint64_t offset = (size + alignment - 1) / alignment * alignment;
        const uint64_t bytes = info.levels[firstMip+i].decodedBytes;
        texture.mipUploads.push_back({offset,std::max(desc.width >> i,1u),std::max(desc.height >> i,1u),bytes});
        size = offset + bytes;
    }
    auto phaseBegin = SceneResourceLogClock::now();
    Result result = createSharedTexture(device, desc, texture.texture);
    if (!result) { log = "Cannot allocate KTX2 texture: " + info.path.string(); return result; }
    result = createSharedTextureView(device, *texture.texture,
        {.format=desc.format,.mipCount=desc.mipCount,.swizzle=info.swizzle}, texture.view);
    timing.imageCreateMs += sceneResourceElapsedMilliseconds(phaseBegin);
    if (!result) { return result; }
    void* mapped = nullptr;
    phaseBegin = SceneResourceLogClock::now();
    result = arena.allocate(device,size,alignment,texture.uploadBuffer,texture.uploadBufferOffset,mapped,log,"KTX2 mip tail");
    timing.stagingMs += sceneResourceElapsedMilliseconds(phaseBegin);
    if (!result) { return result; }
    const auto before = reader.stats();
    if (!prefetched && !reader.open(info, log)) { return makeError(Error::Failure); }
    for (uint32_t i = 0; i < desc.mipCount; ++i) {
        if (!prefetched && !reader.decode(firstMip+i,bytes,log)) { reader.close(); return makeError(Error::Failure); }
        const auto& source = prefetched ? prefetched->mips[i] : bytes;
        UploadCpuTimer copyTimer{timing.copyMs};
        std::memcpy(static_cast<uint8_t*>(mapped)+texture.mipUploads[i].bufferOffset,source.data(),source.size());
    }
    reader.close();
    timing.openMs = prefetched ? prefetched->stats.openMs : reader.stats().openMs - before.openMs;
    timing.readMs = prefetched ? prefetched->stats.readMs : reader.stats().readMs - before.readMs;
    timing.decodeMs = prefetched ? prefetched->stats.decodeMs : reader.stats().decodeMs - before.decodeMs;
    if (prefetched) { timing.totalMs += prefetched->workerMs; }
    phaseBegin = SceneResourceLogClock::now();
    texture.uploadBuffer->flush(texture.uploadBufferOffset,size);
    timing.flushMs += sceneResourceElapsedMilliseconds(phaseBegin);
    texture.uploadAllocationSize = size;
    return {};
}

void stampTextureFormats(std::vector<ScenePathTraceGpuMaterial>& materials,
    const std::vector<ScenePathTraceMaterialTexture>& textures)
{
    for (auto& material : materials) {
        for (auto* info : {&material.baseColorTexture,&material.metallicRoughnessTexture,&material.normalTexture,
             &material.occlusionTexture,&material.emissiveTexture,&material.transmissionTexture,
             &material.thicknessTexture,&material.diffuseTransmissionTexture,&material.diffuseTransmissionColorTexture,&material.specularTexture,&material.specularColorTexture}) {
            const auto index = info->textureIndex;
            if (index >= textures.size()) { continue; }
            // KTX2 vkFormat declares the transfer function: sRGB views decode in
            // hardware, while UNORM KTX2 is already linear. Legacy PNG views
            // retain their existing shader-side sRGB conversion.
            uint32_t flags = compressedBlockBytes(textures[index].format) != 0 ? 1u : 0u;
            if (info == &material.normalTexture && textures[index].format == Format::Bc5Unorm) { flags |= 2u; }
            info->transform0[3] = float(flags);
        }
    }
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
        return 2.0f;
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
    gpuMaterial.emissive[3] = material.unlit ? 1.0f : 0.0f;
    gpuMaterial.specular[0] = material.specularColorFactor.x;
    gpuMaterial.specular[1] = material.specularColorFactor.y;
    gpuMaterial.specular[2] = material.specularColorFactor.z;
    gpuMaterial.specular[3] = material.specularFactor;
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
            "alphaMode BLEND requires the OpenPBR continuation path; legacy shading modes do not composite partial alpha";
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
    gpuMaterial.specularTexture = makeTextureInfo(material.specularTexture, "specularTexture");
    gpuMaterial.specularColorTexture = makeTextureInfo(material.specularColorTexture, "specularColorTexture");
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
    bool includePositions,
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
        outScene.vertices.push_back(SceneShadingVertex{
            .normal = packSceneNormal(normal.x, normal.y, normal.z),
            .tangent = packSceneTangent(tangent.x, tangent.y, tangent.z, tangent.w),
            .texcoord = {texcoord.x, texcoord.y},
        });
        if (includePositions) { outScene.positions.push_back({position.x, position.y, position.z}); }
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
            if (includePositions) { outScene.positions.resize(outPrimitive.firstVertex); }
            outScene.indices.resize(outPrimitive.firstIndex);
            return false;
        }
        outScene.indices.push_back(sourceIndex);
    }
    return true;
}

std::vector<ScenePathTraceGpuMaterial> buildGpuMaterials(
    const scene::Scene& loadedScene,
    const std::vector<uint32_t>& textureIndexMap,
    const std::vector<uint32_t>& neuralTextureSetIndexMap,
    std::string& log)
{
    std::vector<ScenePathTraceGpuMaterial> materials;
    materials.reserve(std::max<size_t>(loadedScene.materials().size(), 1));
    if (loadedScene.materials().empty()) {
        materials.push_back(ScenePathTraceGpuMaterial{});
    } else {
        for (const scene::RenderMaterial& material : loadedScene.materials()) {
            materials.push_back(makeMaterial(
                material,
                loadedScene,
                textureIndexMap,
                neuralTextureSetIndexMap,
                log));
        }
    }

    return materials;
}

bool buildGpuScene(
    const scene::Scene& loadedScene,
    bool includePositions,
    const std::vector<uint32_t>& textureIndexMap,
    const std::vector<uint32_t>& neuralTextureSetIndexMap,
    ScenePathTraceGpuScene& outScene,
    std::string& log,
    bool materialsOnly = false)
{
    outScene = ScenePathTraceGpuScene{};
    outScene.materials = buildGpuMaterials(
        loadedScene, textureIndexMap, neuralTextureSetIndexMap, log);
    if (materialsOnly || loadedScene.hasStreamGeometry()) {
        // Geometry stays in the Streamer page pool. Deferred only needs the
        // same OpenPBR material table and textures as resident shading.
        return !outScene.materials.empty();
    }

    constexpr uint32_t kInvalidPrimitiveIndex = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> primitiveToGpuPrimitive(
        loadedScene.renderPrimitives().size(),
        kInvalidPrimitiveIndex);
    for (uint32_t primitiveIndex = 0; primitiveIndex < loadedScene.renderPrimitives().size(); ++primitiveIndex) {
        ScenePathTraceGpuPrimitive gpuPrimitive;
        if (!appendPrimitiveGeometry(loadedScene.renderPrimitives()[primitiveIndex], includePositions, outScene, gpuPrimitive)) {
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
    std::vector<bool> referenced(loadedScene.textures().size(), loadedScene.hasStreamGeometry());
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
        mark(material.specularTexture);
        mark(material.specularColorTexture);
    }
    return referenced;
}

std::vector<std::array<int32_t, 21>> materialResourceLayout(const scene::Scene& loadedScene)
{
    std::vector<std::array<int32_t, 21>> layout;
    layout.reserve(loadedScene.materials().size());
    for (const scene::RenderMaterial& material : loadedScene.materials()) {
        // Opacity is baked into OMM. Changes to alpha, cutoff, and UV sampling
        // require a BLAS/OMM rebuild; color/roughness still update only the buffer.
        layout.push_back({
            material.baseColorTexture.textureIndex,
            material.metallicRoughnessTexture.textureIndex,
            material.normalTexture.textureIndex,
            material.occlusionTexture.textureIndex,
            material.emissiveTexture.textureIndex,
            material.transmissionTexture.textureIndex,
            material.thicknessTexture.textureIndex,
            material.diffuseTransmissionTexture.textureIndex,
            material.diffuseTransmissionColorTexture.textureIndex,
            material.specularTexture.textureIndex,
            material.specularColorTexture.textureIndex,
            material.alphaMode == "MASK" ? 1 : material.alphaMode == "BLEND" ? 2 : 0,
            std::bit_cast<int32_t>(material.alphaMode == "OPAQUE" ? 1.0f : material.baseColorFactor.w),
            std::bit_cast<int32_t>(material.alphaMode == "MASK" ? material.alphaCutoff : 0.0f),
            material.baseColorTexture.texCoord,
            std::bit_cast<int32_t>(material.baseColorTexture.uvTransform[0]),
            std::bit_cast<int32_t>(material.baseColorTexture.uvTransform[1]),
            std::bit_cast<int32_t>(material.baseColorTexture.uvTransform[2]),
            std::bit_cast<int32_t>(material.baseColorTexture.uvTransform[3]),
            std::bit_cast<int32_t>(material.baseColorTexture.uvTransform[4]),
            std::bit_cast<int32_t>(material.baseColorTexture.uvTransform[5]),
        });
    }
    return layout;
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
        SubmitUploads,
        WaitForGpu,
        Ready,
        Failed,
    };

    static constexpr uint64_t kUploadBatchByteLimit = 64ull * 1024ull * 1024ull;
    static constexpr uint32_t kUploadBatchRegionLimit = 128;
    static constexpr uint32_t kMaxUploadBatchesInFlight = 3;

    struct UploadBatch {
        SceneUploadStagingArena staging;
        std::unique_ptr<CommandPool> uploadPool;
        std::unique_ptr<CommandBuffer> uploadCommands;
        std::unique_ptr<CommandPool> acquirePool;
        std::unique_ptr<CommandBuffer> acquireCommands;
        std::unique_ptr<Semaphore> timeline;
        uint64_t completionValue = 0;
        bool includesNeuralUploads = false;
        SceneResourceLogClock::time_point copySubmitted{}, acquireSubmitted{};
        bool copyObserved = false, acquireObserved = false;
        uint64_t observedCompletionValue = 0;

        bool complete() const
        {
            return completionValue == 0 || (timeline != nullptr && timeline->currentValue() >= completionValue);
        }
    };

    // Physical images are versioned independently of the stable material IDs.
    // Every submitting frame retains its immutable generation, including views.
    struct TextureImageOwner {
        std::shared_ptr<Texture> texture;
        std::shared_ptr<TextureView> view; // Destroy the view before its image.
    };
    using TextureGeneration = std::vector<TextureImageOwner>;
    std::shared_ptr<TextureGeneration> textureGeneration;
    GpuCompletionPoint texturePublication;
    struct TextureFeedback {
        std::shared_ptr<Buffer> buffer;
        GpuCompletionPoint completion;
        uint64_t frame = 0;
    };
    struct RetiredTexture { std::weak_ptr<Texture> texture; uint64_t bytes = 0; };
    struct TextureRequest { uint32_t image = 0, slot = 0, mip = 0; uint64_t bytes = 0; };
    struct TextureMigration {
        std::vector<TextureRequest> requests;
        std::vector<ScenePathTraceMaterialTexture> textures;
        std::unique_ptr<Ktx2TexturePrefetch> decode;
        SceneUploadStagingArena staging;
        Ktx2MipReader reader;
        std::vector<uint8_t> scratch;
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        RenderFrameContext frame;
        QueueSubmissionTracker tracker;
        bool submitted = false;
        uint64_t requestedFrame = 0, submittedFrame = 0;
        SceneResourceLogClock::time_point submittedAt;
        std::unique_ptr<TimestampQueryPool> timestamps;
        ~TextureMigration()
        {
            if (submitted) { (void)frame.wait(); }
            commands.reset();
            (void)frame.reset();
        }
    };
    std::unique_ptr<TextureMigration> textureMigration;
    std::vector<TextureFeedback> textureFeedback;
    std::vector<std::shared_ptr<Buffer>> freeTextureFeedback;
    std::vector<RetiredTexture> retiredTextures;
    std::shared_ptr<Buffer> emptyTextureFeedback;
    std::vector<bool> pinnedImages;
    std::vector<uint32_t> baseTextureMips, desiredTextureMips, textureHits;
    std::vector<uint64_t> textureLastSeen, textureLastChanged;
    uint64_t streamingFrame = 0, lastFeedbackFrame = 0, nextTextureRetry = 0;

    void refreshTextureGeneration()
    {
        auto generation = std::make_shared<TextureGeneration>();
        generation->reserve(materialTextures.size());
        for (size_t slot = 0; slot < materialTextures.size(); ++slot) {
            generation->push_back({materialTextures[slot].texture,materialTextures[slot].view});
            materialTextureViews[slot] = materialTextures[slot].view.get();
        }
        textureGeneration = std::move(generation);
    }

    void initializeTextureStreaming()
    {
        baseTextureMips = desiredTextureMips = ktxFirstMips;
        textureHits.assign(ktxImages.size(), 0);
        textureLastSeen.assign(ktxImages.size(), 0);
        textureLastChanged.assign(ktxImages.size(), 0);
        textureStats.streamingEnabled = textureStreaming;
        textureStats.peakLiveAllocationBytes = textureStats.residentAllocationBytes;
        refreshTextureGeneration();
    }

    void resetTextureStreaming()
    {
        textureMigration.reset(); // Cancel/join CPU work and wait only during teardown.
        textureFeedback.clear(); freeTextureFeedback.clear();
        retiredTextures.clear();
        emptyTextureFeedback.reset();
        textureGeneration.reset();
        texturePublication = {};
        baseTextureMips.clear(); desiredTextureMips.clear(); textureHits.clear();
        textureLastSeen.clear(); textureLastChanged.clear(); pinnedImages.clear();
        streamingFrame = lastFeedbackFrame = nextTextureRetry = 0;
    }

    void consumeTextureFeedback()
    {
        for (auto it = textureFeedback.begin(); it != textureFeedback.end();) {
            if (it->completion.isCancelled()) { it = textureFeedback.erase(it); continue; }
            if (!it->completion.isComplete()) { ++it; continue; }
            if (it->frame >= lastFeedbackFrame) {
                it->buffer->invalidate();
                auto* words = static_cast<const uint32_t*>(it->buffer->map());
                if (words) {
                    for (uint32_t image = 0; image < ktxImages.size(); ++image) {
                        const uint32_t slot = asyncImageTextureIndexMap[image];
                        if (!ktxImages[image] || slot == kInvalidMaterialTextureIndex || pinnedImages[image]) { continue; }
                        const uint32_t mip = words[slot * 8 + 4], hits = words[slot * 8 + 5];
                        if (mip != UINT32_MAX && hits) {
                            const uint32_t finest = ktxImages[image]->firstMipForDimension(textureRefineDimension);
                            desiredTextureMips[image] = std::clamp(mip, std::min(finest, baseTextureMips[image]), baseTextureMips[image]);
                            textureLastSeen[image] = it->frame;
                            textureHits[image] = hits;
                        }
                    }
                    it->buffer->unmap();
                    ++textureStats.feedbackFrames;
                    lastFeedbackFrame = it->frame;
                }
            }
            freeTextureFeedback.push_back(std::move(it->buffer));
            it = textureFeedback.erase(it);
        }
        std::erase_if(retiredTextures, [](const auto& retired) { return retired.texture.expired(); });
        textureStats.retiredAllocationBytes = 0;
        for (const auto& retired : retiredTextures) { textureStats.retiredAllocationBytes += retired.bytes; }
    }

    Result pumpTextureMigration(CpuProfileRecorder* profiler)
    {
        CpuProfileScope phase(profiler, "Poll migration completion");
        if (!textureMigration) { return {}; }
        auto& migration = *textureMigration;
        if (migration.submitted) {
            if (!migration.frame.completion().isComplete()) { return {}; }
            phase.next("Resolve upload timestamps");
            TextureUploadProfile upload;
            upload.sequence = textureStats.lastUpload.sequence + 1;
            upload.requestFrame = migration.requestedFrame;
            upload.submitFrame = migration.submittedFrame;
            upload.completionFrame = streamingFrame;
            upload.images = uint32_t(migration.textures.size());
            for (const auto& image : migration.textures) { upload.bytes += image.byteSize; }
            upload.completionObservedMilliseconds = sceneResourceElapsedMilliseconds(migration.submittedAt);
            if (migration.timestamps) {
                TimestampQueryResult values[2];
                const auto queryResult = migration.timestamps->readResults(0, 2, values);
                if (queryResult && values[0].available && values[1].available) {
                    upload.gpuTimingAvailable = true;
                    upload.gpuMilliseconds = migration.timestamps->durationMilliseconds(values[0].value, values[1].value);
                }
            }
            textureStats.lastUpload = upload;
            phase.next("Publish texture generation");
            texturePublication = migration.frame.completion();
            for (size_t i = 0; i < migration.requests.size(); ++i) {
                const auto& request = migration.requests[i];
                auto& old = materialTextures[request.slot];
                retiredTextures.push_back({old.texture, old.texture->allocationSize()});
                textureStats.residentAllocationBytes -= old.texture->allocationSize();
                textureStats.residentPayloadBytes -= old.byteSize;
                textureStats.residentAllocationBytes += migration.textures[i].texture->allocationSize();
                textureStats.residentPayloadBytes += migration.textures[i].byteSize;
                if (request.mip < ktxFirstMips[request.image]) { ++textureStats.upgrades; }
                else { ++textureStats.downgrades; }
                old = std::move(migration.textures[i]);
                old.uploadBuffer.reset(); old.uploadAllocationSize = 0;
                ktxFirstMips[request.image] = request.mip;
                textureLastChanged[request.image] = streamingFrame;
            }
            textureStats.maxRequestLatencyFrames = std::max(textureStats.maxRequestLatencyFrames,
                streamingFrame - migration.requestedFrame);
            refreshTextureGeneration();
            textureMigration.reset();
            textureStats.pendingImages = 0; textureStats.pendingAllocationBytes = 0;
            return {};
        }
        phase.next("Prepare images and staging");
        const auto prepareStarted = SceneResourceLogClock::now();
        Result result;
        // Consume ready jobs only; cap image count/bytes at scheduling and spend
        // at most one millisecond preparing a batch per tick (one image may overrun).
        while (migration.textures.size() != migration.requests.size()) {
            // Ordered prefetch consumption is nonblocking; at most eight small tails.
            const auto* decoded = migration.decode->front();
            if (!decoded) { return {}; }
            if (!decoded->error.empty()) { return makeError(Error::Failure); }
            const auto& request = migration.requests[migration.textures.size()];
            const auto budget = device->memoryBudget();
            if (budget.policy.enabled && request.bytes > budget.availableBytes) { return makeError(Error::OutOfMemory); }
            ScenePathTraceMaterialTexture texture;
            SceneTextureLoadTiming timing;
            std::string log;
            result = createKtxMaterialTexture(*device, migration.staging, migration.reader, migration.scratch,
                timing, *ktxImages[request.image], request.mip, texture, log, decoded);
            if (!result) { return result; }
            // Validate the actual requirement as well as the pre-allocation estimate.
            if (texture.texture->allocationSize() > request.bytes) { return makeError(Error::OutOfMemory); }
            migration.textures.push_back(std::move(texture));
            migration.decode->pop();
            if (migration.textures.size() != migration.requests.size() &&
                sceneResourceElapsedMilliseconds(prepareStarted) >= 1.0) { return {}; }
        }
        phase.next("Release decode workers");
        migration.decode.reset();
        phase.next("Upload command setup");
        if (!(result = device->createCommandPool(*graphicsQueue, migration.pool)) ||
            !(result = migration.pool->createCommandBuffer(migration.commands)) ||
            !(result = migration.tracker.initialize(*device, *graphicsQueue)) ||
            !(result = migration.frame.begin(streamingFrame)) ||
            !(result = migration.commands->begin(&migration.frame))) { return result; }
        if (graphicsQueue->timestampValidBits() != 0) {
            if (device->createTimestampQueryPool(*graphicsQueue, {.queryCount=2}, migration.timestamps)) {
                if (!(result = migration.commands->resetTimestampQueries(*migration.timestamps, 0, 2)) ||
                    !(result = migration.commands->writeTimestamp(*migration.timestamps, 0, PipelineStageBits::TopOfPipe))) { return result; }
            }
        }
        phase.next("Record texture copies");
        for (auto& image : migration.textures) {
            if (!(result = uploadTexture(*migration.commands, image))) { return result; }
            TextureBarrierDesc barrier{.texture=image.texture.get(), .before=ResourceState::TransferDestination,
                .after=ResourceState::ShaderRead, .mipCount=image.mipCount, .layerCount=1};
            migration.commands->barrier({.textures=&barrier, .textureCount=1});
            image.state = ResourceState::ShaderRead;
        }
        if (migration.timestamps && !(result = migration.commands->writeTimestamp(
            *migration.timestamps, 1, PipelineStageBits::BottomOfPipe))) { return result; }
        if (!(result = migration.commands->end())) { return result; }
        phase.next("Submit texture upload");
        CommandBuffer* commands[] = {migration.commands.get()};
        if (!(result = migration.tracker.submit({.commandBuffers=commands, .commandBufferCount=1}, migration.frame))) { return result; }
        migration.submitted = true;
        migration.submittedFrame = streamingFrame;
        migration.submittedAt = SceneResourceLogClock::now();
        for (const auto& image : migration.textures) { textureStats.streamingUploadBytes += image.byteSize; }
        return {};
    }

    void scheduleTextureMigration(CpuProfileRecorder* profiler)
    {
        if (textureMigration || streamingFrame < nextTextureRetry) { return; }
        constexpr uint64_t maxBatchBytes = 4ull * 1024 * 1024;
        // Keep real headroom for replacement tails; old + new coexist until retire.
        const uint64_t reserve = std::min<uint64_t>(16ull * 1024 * 1024, textureStats.budgetBytes / 8);
        const uint64_t steadyBudget = textureStats.budgetBytes - reserve;
        CpuProfileScope phase(profiler, "Candidates and allocation queries");
        const auto budget = device->memoryBudget();
        const bool pressure = textureStats.residentAllocationBytes > steadyBudget ||
            (budget.policy.enabled && budget.availableBytes < reserve);
        struct Candidate { TextureRequest request; double priority; };
        std::vector<Candidate> candidates;
        textureStats.refinedImages = textureStats.requestedImages = 0;
        for (uint32_t image = 0; image < ktxImages.size(); ++image) {
            if (!ktxImages[image] || pinnedImages[image]) { continue; }
            const uint32_t slot = asyncImageTextureIndexMap[image];
            if (slot == kInvalidMaterialTextureIndex) { continue; }
            const uint32_t current = ktxFirstMips[image], base = baseTextureMips[image];
            textureStats.refinedImages += current < base;
            const uint64_t age = streamingFrame - textureLastSeen[image];
            uint32_t target = current;
            const bool cold = age >= textureColdFrames || (pressure && age >= std::min(30u, textureColdFrames));
            if (cold && current < base) { target = base; }
            else if (age < 16 && desiredTextureMips[image] < current) {
                ++textureStats.requestedImages;
                if (!pressure) { target = std::max(desiredTextureMips[image], current - 1); }
                else { ++textureStats.budgetDeferrals; }
            }
            // A separate cooldown prevents refine/evict oscillation around a view cut.
            if (target == current || streamingFrame - textureLastChanged[image] < std::min(30u, textureColdFrames)) { continue; }
            uint64_t bytes = 0;
            if (!device->textureAllocationSize(ktxImages[image]->textureDesc(target), bytes)) { continue; }
            const uint64_t oldBytes = materialTextures[slot].texture->allocationSize();
            const double priority = target > current ? 1e12 + double(oldBytes) :
                double(textureHits[image]) * (current - target) / double(std::max<uint64_t>(bytes - std::min(bytes,oldBytes),1));
            candidates.push_back({{image,slot,target,bytes},priority});
        }
        phase.next("Sort and admit candidates");
        std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.priority > b.priority; });
        auto migration = std::make_unique<TextureMigration>();
        uint64_t bytes = 0, cpuBytes = 0;
        uint64_t projected = textureStats.residentAllocationBytes;
        const uint64_t live = projected + textureStats.retiredAllocationBytes;
        std::vector<Ktx2PrefetchRequest> decode;
        for (const auto& candidate : candidates) {
            const auto& request = candidate.request;
            const uint64_t workingBytes = Ktx2TexturePrefetch::requiredBytes(*ktxImages[request.image], request.mip);
            if (request.bytes > maxBatchBytes || bytes + request.bytes > maxBatchBytes || cpuBytes + workingBytes > kTexturePrefetchBytes) { continue; }
            const uint64_t after = projected - materialTextures[request.slot].texture->allocationSize() + request.bytes;
            const bool upgrade = request.mip < ktxFirstMips[request.image];
            if (live + bytes + request.bytes > textureStats.budgetBytes || (upgrade && after > steadyBudget) ||
                (budget.policy.enabled && bytes + request.bytes > budget.availableBytes)) { ++textureStats.budgetDeferrals; continue; }
            bytes += request.bytes; cpuBytes += workingBytes; projected = after;
            migration->requests.push_back(request);
            decode.push_back({*ktxImages[request.image],request.mip});
            if (decode.size() == 8) { break; }
        }
        if (decode.empty()) { return; }
        phase.next("Start decode workers");
        migration->decode = std::make_unique<Ktx2TexturePrefetch>(std::move(decode), std::min(textureLoadWorkers,2u), kTexturePrefetchBytes);
        migration->requestedFrame = streamingFrame;
        textureStats.pendingImages = uint32_t(migration->requests.size());
        textureStats.pendingAllocationBytes = bytes;
        textureStats.peakLiveAllocationBytes = std::max(textureStats.peakLiveAllocationBytes,live + bytes);
        textureMigration = std::move(migration);
    }

    Result beginTextureStreaming(CommandBuffer& commands, uint64_t frameIndex, Buffer*& feedback, CpuProfileRecorder* profiler)
    {
        if (!emptyTextureFeedback) {
            std::unique_ptr<Buffer> buffer;
            auto result = device->createBuffer({.size=32, .usage=BufferUsageBits::Storage,
                .memoryLocation=MemoryLocation::HostUpload}, buffer);
            if (!result) { return result; }
            auto* mapped = buffer->map(); if (!mapped) { return makeError(Error::Failure); }
            std::memset(mapped,0,32); buffer->flush(); buffer->unmap();
            emptyTextureFeedback = std::move(buffer);
        }
        feedback = emptyTextureFeedback.get();
        auto* frame = commands.frameContext();
        if (!frame) { return {}; }
        frame->retain(emptyTextureFeedback);
        if (!textureStreaming || baseTextureMips.empty()) { return {}; }
        CpuProfileScope phase(profiler, "Consume texture feedback");
        streamingFrame = frameIndex;
        consumeTextureFeedback();
        phase.next("Texture migration");
        auto result = pumpTextureMigration(profiler);
        if (!result) {
            spdlog::warn("[TextureStreaming] Migration deferred: {}", resultToString(result));
            textureMigration.reset();
            textureStats.pendingImages = 0; textureStats.pendingAllocationBytes = 0;
            nextTextureRetry = streamingFrame + 60;
            ++textureStats.budgetDeferrals;
            if (hasError(result,Error::DeviceLost)) { return result; }
        }
        // Publication can release the previous generation immediately if no frame uses it.
        phase.next("Retire texture generations");
        consumeTextureFeedback();
        phase.next("Schedule texture migration");
        scheduleTextureMigration(profiler);
        phase.next("Prepare texture feedback");
        if (textureFeedback.size() >= 4) { return {}; }
        std::shared_ptr<Buffer> buffer;
        if (!freeTextureFeedback.empty()) {
            buffer = std::move(freeTextureFeedback.back()); freeTextureFeedback.pop_back();
        } else {
            std::unique_ptr<Buffer> created;
            result = device->createBuffer({.size=materialTextures.size()*32, .usage=BufferUsageBits::Storage,
                .memoryLocation=MemoryLocation::HostReadback},created);
            if (!result) { return result; }
            buffer = std::move(created);
        }
        auto* words = static_cast<uint32_t*>(buffer->map());
        if (!words) { return makeError(Error::Failure); }
        std::memset(words,0,materialTextures.size()*32);
        for (uint32_t image = 0; image < ktxImages.size(); ++image) {
            const uint32_t slot = asyncImageTextureIndexMap[image];
            if (!ktxImages[image] || pinnedImages[image] || slot == kInvalidMaterialTextureIndex) { continue; }
            words[slot*8] = ktxImages[image]->width;
            words[slot*8+1] = ktxImages[image]->height;
            words[slot*8+2] = uint32_t(ktxImages[image]->levels.size());
            words[slot*8+3] = ktxFirstMips[image];
            words[slot*8+4] = UINT32_MAX;
        }
        buffer->flush(); buffer->unmap();
        TextureFeedback entry{std::move(buffer),frame->completion(),frameIndex};
        feedback = entry.buffer.get(); frame->retain(entry.buffer);
        BufferBarrierDesc ready{.buffer=feedback, .before=ResourceState::Undefined, .after=ResourceState::General};
        commands.barrier({.buffers=&ready, .bufferCount=1});
        textureFeedback.push_back(std::move(entry));
        return {};
    }

    uint64_t pendingUploadByteSize() const
    {
        uint64_t byteSize = neuralTextures.pendingUploadByteSize() + pendingTextureBytes;
        for (const ScenePathTraceBufferUpload& upload : bufferUploads) {
            byteSize += upload.byteSize;
        }
        return byteSize;
    }

    uint32_t pendingUploadRegionCount() const
    {
        uint64_t regionCount = bufferUploads.size() +
            neuralTextures.pendingUploadRegionCount() + pendingTextureRegions;
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
        if (imageIndex < ktxImages.size() && ktxImages[imageIndex]) {
            const auto& info = *ktxImages[imageIndex];
            return info.tailBytes(ktxFirstMips[imageIndex]) + info.levels.size() * 256;
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
        if (imageIndex < ktxImages.size() && ktxImages[imageIndex]) {
            return uint32_t(ktxImages[imageIndex]->levels.size()) - ktxFirstMips[imageIndex];
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
        if (batchRegionCount == 0) {
            return {};
        }
        // Keep submitted command buffers, timelines and staging alive even if
        // creating or submitting the graphics acquire step subsequently fails.
        auto& batch = uploadBatches.emplace_back(std::make_unique<UploadBatch>());
        batch->staging = stagingArena.takeUsedPages();
        batch->includesNeuralUploads = neuralTextures.pendingUploadRegionCount() != 0;
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

        auto phaseBegin = SceneResourceLogClock::now();
        Result result = device.createCommandPool(*uploadQueue, batch->uploadPool);
        if (!result || batch->uploadPool == nullptr) {
            log += resultMessage("createCommandPool(scene texture uploads)", result);
            return result ? makeError(Error::Failure) : result;
        }
        result = batch->uploadPool->createCommandBuffer(batch->uploadCommands);
        if (!result || batch->uploadCommands == nullptr) {
            log += resultMessage("createCommandBuffer(scene texture uploads)", result);
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createSemaphore(
            SemaphoreDesc{.initialValue = 0},
            batch->timeline);
        if (!result || batch->timeline == nullptr) {
            log += resultMessage("createSemaphore(scene uploads)", result);
            return result ? makeError(Error::Failure) : result;
        }
        uploadStats.commandSetupMs += sceneResourceElapsedMilliseconds(phaseBegin);
        phaseBegin = SceneResourceLogClock::now();
        result = batch->uploadCommands->begin();
        if (!result) {
            log += resultMessage("CommandBuffer::begin(scene texture uploads)", result);
            return result;
        }
        for (const auto index : pendingTextures) {
            result = uploadTexture(*batch->uploadCommands, materialTextures[index]);
            if (!result) {
                return result;
            }
        }
        result = neuralTextures.recordUploads(*batch->uploadCommands);
        if (!result) {
            return result;
        }
        for (const ScenePathTraceBufferUpload& upload : bufferUploads) {
            batch->uploadCommands->copyBuffer(BufferCopyDesc{
                .source = upload.stagingBuffer.get(),
                .destination = upload.destination,
                .sourceOffset = upload.sourceOffset,
                .size = upload.byteSize,
            });
        }
        if (!requiresGraphicsAcquire) {
            transitionUploadsForRendering(*batch->uploadCommands);
        }
        result = batch->uploadCommands->end();
        if (!result) {
            log += resultMessage("CommandBuffer::end(scene texture uploads)", result);
            return result;
        }

        CommandBuffer* commandBuffers[] = {batch->uploadCommands.get()};
        const SemaphoreSubmitDesc signal{
            .semaphore = batch->timeline.get(),
            .value = 1,
            .stages = requiresGraphicsAcquire ? PipelineStageBits::Transfer : PipelineStageBits::AllCommands,
        };
        uploadStats.recordMs += sceneResourceElapsedMilliseconds(phaseBegin);
        batch->copySubmitted = SceneResourceLogClock::now();
        result = uploadQueue->submit(QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalSemaphores = &signal,
            .signalSemaphoreCount = 1,
        });
        uploadStats.copySubmitMs += sceneResourceElapsedMilliseconds(batch->copySubmitted);
        if (!result) {
            log += resultMessage("Queue::submit(scene texture uploads)", result);
            return result;
        }

        batch->completionValue = 1;
        ++uploadStats.submittedBatches;
        uploadStats.submittedBytes += batchByteSize;
        uploadStats.inFlightBatches = static_cast<uint32_t>(uploadBatches.size());
        uploadStats.peakInFlightBatches = std::max(uploadStats.peakInFlightBatches, uploadStats.inFlightBatches);
        if (requiresGraphicsAcquire) {
            phaseBegin = SceneResourceLogClock::now();
            result = device.createCommandPool(graphicsQueue, batch->acquirePool);
            if (!result || batch->acquirePool == nullptr) {
                log += resultMessage("createCommandPool(scene upload acquire)", result);
                return result ? makeError(Error::Failure) : result;
            }
            result = batch->acquirePool->createCommandBuffer(batch->acquireCommands);
            if (!result || batch->acquireCommands == nullptr) {
                log += resultMessage("createCommandBuffer(scene upload acquire)", result);
                return result ? makeError(Error::Failure) : result;
            }
            uploadStats.commandSetupMs += sceneResourceElapsedMilliseconds(phaseBegin);
            phaseBegin = SceneResourceLogClock::now();
            result = batch->acquireCommands->begin();
            if (!result) {
                log += resultMessage("CommandBuffer::begin(scene upload acquire)", result);
                return result;
            }
            transitionUploadsForRendering(*batch->acquireCommands);
            result = batch->acquireCommands->end();
            if (!result) {
                log += resultMessage("CommandBuffer::end(scene upload acquire)", result);
                return result;
            }

            CommandBuffer* acquireCommandBuffers[] = {batch->acquireCommands.get()};
            const SemaphoreSubmitDesc wait{
                .semaphore = batch->timeline.get(),
                .value = 1,
                .stages = PipelineStageBits::AllCommands,
            };
            const SemaphoreSubmitDesc acquireSignal{
                .semaphore = batch->timeline.get(),
                .value = 2,
                .stages = PipelineStageBits::AllCommands,
            };
            uploadStats.recordMs += sceneResourceElapsedMilliseconds(phaseBegin);
            batch->acquireSubmitted = SceneResourceLogClock::now();
            result = graphicsQueue.submit(QueueSubmitDesc{
                .waitSemaphores = &wait,
                .waitSemaphoreCount = 1,
                .commandBuffers = acquireCommandBuffers,
                .commandBufferCount = 1,
                .signalSemaphores = &acquireSignal,
                .signalSemaphoreCount = 1,
            });
            uploadStats.acquireSubmitMs += sceneResourceElapsedMilliseconds(batch->acquireSubmitted);
            if (!result) {
                log += resultMessage("Queue::submit(scene upload acquire)", result);
                return result;
            }
            batch->completionValue = 2;
        }
        // The batch arena owns every staging page until its acquire completes.
        // Remove only this batch's pending uploads so later batches cannot
        // record them again or have their data retired by an earlier batch.
        for (const auto index : pendingTextures) {
            auto& texture = materialTextures[index];
            if (texture.uploaded) {
                texture.uploadBuffer.reset();
            }
        }
        pendingTextures.clear();
        pendingTextureBytes = pendingTextureRegions = 0;
        bufferUploads.clear();
        spdlog::debug(
            "[SceneResources] Submitted scene upload batch bytes={} regions={} queue={} independentCopy={} inFlight={}/{}",
            batchByteSize,
            batchRegionCount,
            uploadQueue->type() == QueueType::Copy ? "copy" : "graphics",
            device.capabilities().independentCopyQueue,
            uploadBatches.size(), kMaxUploadBatchesInFlight);
        if (sceneResourceElapsedMilliseconds(lastUploadProgress) >= 1000) {
            spdlog::info("[SceneResources] Upload progress textures={}/{} batches={} bytes={} inFlight={}/{} open/read/decode={:.1f}/{:.1f}/{:.1f} ms",
                asyncTextureCursor, asyncReferencedTextures.size(), uploadStats.submittedBatches,
                uploadStats.submittedBytes, uploadBatches.size(), kMaxUploadBatchesInFlight,
                uploadStats.ktx.openMs, uploadStats.ktx.readMs, uploadStats.ktx.decodeMs);
            lastUploadProgress = SceneResourceLogClock::now();
            device.logMemoryBudget("texture upload progress");
        }
        return {};
    }

    bool textureUploadsReady()
    {
        retireCompletedTextureUploads();
        const bool ready = uploadBatches.empty();
        if (ready && prepared) { finishUploadProfile(); }
        return ready;
    }

    void retireCompletedTextureUploads()
    {
        for (auto& batch : uploadBatches) {
            if (batch->completionValue == 0 || !batch->timeline) { continue; }
            const auto value = batch->timeline->currentValue();
            batch->observedCompletionValue = value;
            if (value >= 1 && !batch->copyObserved) {
                const auto ms = sceneResourceElapsedMilliseconds(batch->copySubmitted);
                uploadStats.copyCompletionObservedMs += ms;
                uploadStats.maxCopyCompletionObservedMs = std::max(uploadStats.maxCopyCompletionObservedMs, ms);
                ++uploadStats.copyCompletionSamples;
                batch->copyObserved = true;
            }
            if (value >= 2 && batch->completionValue == 2 && !batch->acquireObserved) {
                const auto ms = sceneResourceElapsedMilliseconds(batch->acquireSubmitted);
                uploadStats.acquireCompletionObservedMs += ms;
                uploadStats.maxAcquireCompletionObservedMs = std::max(uploadStats.maxAcquireCompletionObservedMs, ms);
                ++uploadStats.acquireCompletionSamples;
                batch->acquireObserved = true;
            }
        }
        // Retire from the same observation used for timing. A second counter
        // query could complete acquire between queries and lose that sample.
        while (!uploadBatches.empty() && (uploadBatches.front()->completionValue == 0 ||
            uploadBatches.front()->observedCompletionValue >= uploadBatches.front()->completionValue)) {
            auto& batch = *uploadBatches.front();
            if (batch.includesNeuralUploads) {
                neuralTextures.releaseUploadBuffers();
            }
            stagingArena.recycle(std::move(batch.staging));
            if (batch.completionValue != 0) {
                ++uploadStats.completedBatches;
            }
            uploadBatches.pop_front();
        }
        uploadStats.inFlightBatches = static_cast<uint32_t>(uploadBatches.size());
    }

    void finishUploadProfile()
    {
        if (uploadProfileReported) { return; }
        uploadProfileReported = true;
        uploadStats.textureWallMs = sceneResourceElapsedMilliseconds(textureLoadBegin);
        if (device) { device->logMemoryBudget("texture uploads completed"); }
        spdlog::info("[SceneResources] Upload profile wall={:.2f} header={:.2f} plan={:.2f} build={:.2f} open={:.2f} read={:.2f} decode={:.2f} image={:.2f} staging={:.2f} memcpy={:.2f} flush={:.2f} ms",
            uploadStats.textureWallMs, uploadStats.textureHeaderMs, uploadStats.texturePlanMs, uploadStats.textureBuildMs,
            uploadStats.ktx.openMs, uploadStats.ktx.readMs, uploadStats.ktx.decodeMs, uploadStats.imageCreateMs,
            uploadStats.stagingMs, uploadStats.stagingCopyMs, uploadStats.stagingFlushMs);
        spdlog::info("[SceneResources] Upload profile setup={:.2f} record={:.2f} copySubmit={:.2f} acquireSubmit={:.2f} backpressure={:.2f} finalWait={:.2f} ms; opens={} mips={} decoders={} storedBytes={} decodedBytes={}",
            uploadStats.commandSetupMs, uploadStats.recordMs, uploadStats.copySubmitMs, uploadStats.acquireSubmitMs,
            uploadStats.backpressureMs, uploadStats.finalWaitMs, uploadStats.ktx.fileOpens, uploadStats.ktx.decodedMips,
            uploadStats.ktx.decoderCreations, uploadStats.ktx.storedBytes, uploadStats.ktx.decodedBytes);
        spdlog::info("[SceneResources] Host-observed completion latency (NOT GPU time): copy avg/max={:.2f}/{:.2f} acquire avg/max={:.2f}/{:.2f} ms",
            uploadStats.copyCompletionObservedMs / std::max<uint64_t>(1, uploadStats.copyCompletionSamples), uploadStats.maxCopyCompletionObservedMs,
            uploadStats.acquireCompletionObservedMs / std::max<uint64_t>(1, uploadStats.acquireCompletionSamples), uploadStats.maxAcquireCompletionObservedMs);
        spdlog::info("[SceneResources] CPU prefetch workers={} peakJobs={} peakBytes={}/{} decodeWait={:.2f} ms; build/open/read/decode are accumulated work, not wall time",
            uploadStats.prefetch.workers, uploadStats.prefetch.peakJobs, uploadStats.prefetch.peakBytes,
            uploadStats.prefetch.byteLimit, uploadStats.decodeWaitMs);
        for (const auto& t : uploadStats.slowestTextures) {
            spdlog::info("[SceneResources] Slow texture total={:.2f} open/read/decode={:.2f}/{:.2f}/{:.2f} image={:.2f} staging/copy/flush={:.2f}/{:.2f}/{:.2f} ms path='{}'",
                t.totalMs, t.openMs, t.readMs, t.decodeMs, t.imageCreateMs, t.stagingMs, t.copyMs, t.flushMs, t.path);
        }
    }

    void waitForTextureUploads()
    {
        for (const auto& batch : uploadBatches) {
            if (!batch->complete()) {
                (void)batch->timeline->wait(batch->completionValue);
            }
        }
        retireCompletedTextureUploads();
    }

    Result planTextureResources(Device& device, const scene::Scene& scene, std::string& log)
    {
        auto phaseBegin = SceneResourceLogClock::now();
        textureStats = {};
        device.logMemoryBudget("texture planning");
        ktxImages.clear(); ktxImages.resize(scene.images().size());
        ktxFirstMips.assign(scene.images().size(),0);
        pinnedImages.assign(scene.images().size(), false);
        const auto referenced = referencedMaterialTextures(scene);
        struct Probe { uint32_t imageIndex; std::filesystem::path path; std::string error; };
        std::vector<Probe> probes;
        std::vector<bool> visited(scene.images().size());
        for (size_t t = 0; t < scene.textures().size(); ++t) {
            if (!referenced[t] || neuralTextures.logicalTextureSetIndex(uint32_t(t)) != kInvalidNeuralTextureSetIndex) { continue; }
            ++textureStats.logicalTextureCount;
            const auto imageIndex = scene.textures()[t].imageIndex;
            if (imageIndex < 0 || size_t(imageIndex) >= scene.images().size()) { continue; }
            if (visited[imageIndex]) { continue; }
            visited[imageIndex] = true;
            const auto& image = scene.images()[imageIndex];
            if (image.uri.empty() || image.uri.starts_with("data:")) { continue; }
            auto path = std::filesystem::path(image.uri);
            if (path.is_relative()) { path = scene.filename().parent_path()/path; }
            probes.push_back({uint32_t(imageIndex), std::move(path), {}});
        }
        std::atomic<size_t> nextProbe = 0;
        std::vector<std::jthread> probeWorkers;
        for (size_t worker = 0; worker < std::min<size_t>(textureLoadWorkers, probes.size()); ++worker) {
            probeWorkers.emplace_back([&] {
                for (size_t i = nextProbe.fetch_add(1); i < probes.size(); i = nextProbe.fetch_add(1)) {
                    auto& probe = probes[i];
                    try {
                        if (!isKtx2File(probe.path) && probe.path.extension() != ".ktx2") { continue; }
                        auto& info = ktxImages[probe.imageIndex];
                        info.emplace();
                        readKtx2TextureInfo(probe.path, *info, probe.error);
                    } catch (const std::exception& exception) { probe.error = exception.what(); }
                }
            });
        }
        probeWorkers.clear(); // Join all probes before allocation planning or error return.
        for (const auto& probe : probes) {
            if (!probe.error.empty()) { log = probe.error; return makeError(Error::InvalidArgument); }
            if (!ktxImages[probe.imageIndex]) { continue; }
            ++textureStats.ktxImageCount;
            const auto& mime = scene.images()[probe.imageIndex].mimeType;
            if (!mime.empty() && mime != "image/ktx2") { ++textureStats.mimeMismatchCount; }
        }
        std::vector<bool> protectedImages(ktxImages.size());
        for (const auto& material : scene.materials()) {
            const auto textureIndex = material.baseColorTexture.textureIndex;
            if (material.alphaMode != "MASK" || textureIndex < 0 || size_t(textureIndex) >= scene.textures().size()) { continue; }
            const auto imageIndex = scene.textures()[textureIndex].imageIndex;
            if (imageIndex >= 0 && size_t(imageIndex) < ktxImages.size() && ktxImages[imageIndex]) {
                protectedImages[imageIndex] = true;
            }
        }
        for (const auto& material : scene.materials()) {
            const auto pin = [&](int32_t texture) {
                if (texture < 0 || size_t(texture) >= scene.textures().size()) { return; }
                const auto image = scene.textures()[texture].imageIndex;
                if (image >= 0 && size_t(image) < pinnedImages.size()) { pinnedImages[image] = true; }
            };
            if (material.alphaMode != "OPAQUE") { pin(material.baseColorTexture.textureIndex); }
            if (material.displacementMagnitude != 0) { pin(material.displacementTexture.textureIndex); }
        }
        textureStats.maskImageCount = uint32_t(std::count(protectedImages.begin(), protectedImages.end(), true));
        textureStats.maskMaxDimension = textureMaskMaxDimension;
        uploadStats.textureHeaderMs += sceneResourceElapsedMilliseconds(phaseBegin);
        UploadCpuTimer planTimer{uploadStats.texturePlanMs};
        const auto sharedBudget = device.memoryBudget();
        // Dedicated images consume more driver heap space than the sum of
        // VkMemoryRequirements (observed with thousands of small BC tails).
        // Keep this conservative allowance separate from actual image bytes.
        const uint64_t imageCount = uint64_t(textureStats.ktxImageCount) + 1;
        const uint64_t overheadPerImage = sharedBudget.policy.enabled ? sharedBudget.policy.materialImageOverheadBytes : 0;
        const uint64_t dedicatedOverhead = overheadPerImage > UINT64_MAX / imageCount
            ? UINT64_MAX : overheadPerImage * imageCount;
        // Small material images share 16 MiB blocks. Leave two blocks for
        // alignment/tail slack; retain the conservative per-image estimate for
        // large-image policies that can exceed a pool block.
        const uint64_t heapOverhead = std::max(textureMaxDimension, textureMaskMaxDimension) <= 1024
            ? std::min(dedicatedOverhead, 32ull * 1024 * 1024) : dedicatedOverhead;
        const uint64_t textureAvailable = sharedBudget.availableBytes - std::min(sharedBudget.availableBytes, heapOverhead);
        const uint64_t effectiveBudget = sharedBudget.policy.enabled
            ? std::min(textureBudgetBytes, textureAvailable) : textureBudgetBytes;
        textureStats.configuredBudgetBytes = textureBudgetBytes;
        textureStats.sharedAvailableBytes = sharedBudget.availableBytes;
        textureStats.plannedHeapOverheadBytes = heapOverhead;
        const TextureDesc fallback{.usage=TextureUsageBits::Sampled|TextureUsageBits::TransferDestination,
            .format=Format::Rgba8Unorm,.queueAccess=QueueAccessBits::Graphics|QueueAccessBits::Copy};
        uint64_t fallbackBytes = 0;
        auto result = device.textureAllocationSize(fallback,fallbackBytes);
        if (!result) { return result; }
        uint32_t cap = textureMaxDimension;
        for (;;) {
            uint64_t allocation = fallbackBytes, payload = 4;
            for (size_t i = 0; i < ktxImages.size(); ++i) {
                if (!ktxImages[i]) { continue; }
                const auto& info = *ktxImages[i];
                const uint32_t first = info.firstMipForDimension(protectedImages[i] && textureMaskMaxDimension != 0
                    ? std::max(cap, textureMaskMaxDimension) : cap);
                uint64_t bytes = 0;
                result = device.textureAllocationSize(info.textureDesc(first),bytes);
                if (!result) { log = "Unsupported texture allocation: " + info.path.string(); return result; }
                allocation += bytes; payload += info.tailBytes(first); ktxFirstMips[i] = first;
            }
            if (allocation <= effectiveBudget) {
                textureStats.plannedAllocationBytes = allocation; textureStats.plannedPayloadBytes = payload;
                textureStats.selectedMaxDimension = cap; textureStats.budgetBytes = effectiveBudget; break;
            }
            if (cap == 1) {
                log = "Unified GPU / material texture budget cannot hold minimum mip tails with MASK quality floor: required=" +
                    std::to_string(allocation) + " available=" + std::to_string(effectiveBudget);
                return makeError(Error::OutOfMemory);
            }
            cap = std::max(cap/2,1u);
        }
        spdlog::info("[SceneTextures] logical={} KTX2={} MIME-mismatch={} cap={} payload={} allocation={} budget={}",
            textureStats.logicalTextureCount,textureStats.ktxImageCount,textureStats.mimeMismatchCount,
            textureStats.selectedMaxDimension,textureStats.plannedPayloadBytes,textureStats.plannedAllocationBytes,effectiveBudget);
        spdlog::info("[SceneTextures] MASK images={} protectedCap={} ordinaryCap={}",
            textureStats.maskImageCount, textureMaskMaxDimension, cap);
        if (effectiveBudget < textureBudgetBytes) {
            spdlog::info("[SceneTextures] Unified budget limits textures configured={} available={} imageHeapAllowance={} effective={} selectedCap={}",
                textureBudgetBytes, sharedBudget.availableBytes, heapOverhead, effectiveBudget, cap);
        }
        return {};
    }

    void trackTexture(const ScenePathTraceMaterialTexture& texture)
    {
        textureStats.residentPayloadBytes += texture.byteSize;
        textureStats.residentAllocationBytes += texture.texture->allocationSize();
        ++textureStats.residentImageCount;
        uint64_t staging = stagingArena.allocatedBytes();
        for (const auto& batch : uploadBatches) { staging += batch->staging.allocatedBytes(); }
        textureStats.peakStagingBytes = std::max(textureStats.peakStagingBytes,staging);
    }

    Result buildMaterialTextures(Device& device, const scene::Scene& loadedScene,
        std::vector<uint32_t>& outTextureIndexMap, std::string& log)
    {
        auto result = beginMaterialTextureBuild(device,loadedScene,log);
        bool complete = false;
        while (result && !complete) {
            retireCompletedTextureUploads();
            if (uploadBatches.size() >= kMaxUploadBatchesInFlight) {
                UploadCpuTimer waitTimer{uploadStats.backpressureMs};
                result = uploadBatches.front()->timeline->wait(uploadBatches.front()->completionValue);
                continue;
            }
            if (shouldFlushBefore(nextMaterialTextureUploadByteSize(loadedScene),nextMaterialTextureUploadRegionCount(loadedScene))) {
                result = submitTextureUploads(device,*graphicsQueue,log); continue;
            }
            result = buildMaterialTextureStep(device,loadedScene,complete,log);
            if (result && textureDecodePending) { ktxPrefetch->wait(); }
            if (result && uploadBatchLimitReached()) { result = submitTextureUploads(device,*graphicsQueue,log); }
        }
        outTextureIndexMap = textureIndexMap;
        if (!result) { ktxPrefetch.reset(); }
        return result;
    }

    Result beginMaterialTextureBuild(
        Device& device,
        const scene::Scene& loadedScene,
        std::string& log)
    {
        textureLoadBegin = SceneResourceLogClock::now();
        lastUploadProgress = textureLoadBegin;
        ktxReader = std::make_unique<Ktx2MipReader>();
        decodedKtxScratch.clear();
        Result result = neuralTextures.prepare(device, loadedScene, log);
        if (!result) {
            return result;
        }
        materialTextures.clear();
        materialTextureViews.clear();
        materialTextureCount = 0;
        textureIndexMap.assign(loadedScene.textures().size(), kInvalidMaterialTextureIndex);
        pendingTextures.clear();
        pendingTextureBytes = pendingTextureRegions = 0;
        asyncImageTextureIndexMap.assign(
            loadedScene.images().size(),
            kInvalidMaterialTextureIndex);
        asyncReferencedTextures = referencedMaterialTextures(loadedScene);
        asyncTextureCursor = 0;
        result = planTextureResources(device,loadedScene,log);
        if (!result) { return result; }

        std::vector<Ktx2PrefetchRequest> requests;
        prefetchedImages.assign(ktxImages.size(), false);
        // Admission follows logical texture order, so a later ready result can
        // never hold all credits ahead of an unscheduled earlier texture.
        for (size_t t = 0; t < loadedScene.textures().size(); ++t) {
            if (!asyncReferencedTextures[t] || neuralTextures.logicalTextureSetIndex(uint32_t(t)) != kInvalidNeuralTextureSetIndex) { continue; }
            const auto image = loadedScene.textures()[t].imageIndex;
            if (image < 0 || size_t(image) >= ktxImages.size() || !ktxImages[image] || prefetchedImages[image]) { continue; }
            if (Ktx2TexturePrefetch::requiredBytes(*ktxImages[image], ktxFirstMips[image]) > kTexturePrefetchBytes) { continue; }
            prefetchedImages[image] = true;
            requests.push_back({*ktxImages[image], ktxFirstMips[image]});
        }
        ktxPrefetch = std::make_unique<Ktx2TexturePrefetch>(std::move(requests), textureLoadWorkers, kTexturePrefetchBytes);

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
        trackTexture(fallbackTexture);
        pendingTextures.push_back(0);
        pendingTextureBytes += fallbackTexture.uploadAllocationSize;
        pendingTextureRegions += fallbackTexture.mipUploads.size();
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
        textureDecodePending = false;
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
            if (device.capabilities().maxBindlessSampledImages != 0 &&
                materialTextures.size() + 32 >= device.capabilities().maxBindlessSampledImages) {
                log = "ScenePathTracePass exceeded the material texture descriptor limit";
                return makeError(Error::Unsupported);
            }

            ScenePathTraceMaterialTexture materialTexture;
            Result result;
            const size_t imageIndex = logicalTexture.imageIndex < 0 ? SIZE_MAX : size_t(logicalTexture.imageIndex);
            if (imageIndex < ktxImages.size() && ktxImages[imageIndex]) {
                const Ktx2PrefetchResult* prefetched = nullptr;
                if (prefetchedImages[imageIndex]) {
                    prefetched = ktxPrefetch->front();
                    if (!prefetched) {
                        --asyncTextureCursor;
                        textureDecodePending = true;
                        if (!decodeWaitBegin) { decodeWaitBegin = SceneResourceLogClock::now(); }
                        return {};
                    }
                    if (decodeWaitBegin) {
                        uploadStats.decodeWaitMs += sceneResourceElapsedMilliseconds(*decodeWaitBegin);
                        decodeWaitBegin.reset();
                    }
                    if (!prefetched->error.empty()) {
                        log = prefetched->error;
                        ktxPrefetch.reset();
                        return makeError(Error::Failure);
                    }
                }
                SceneTextureLoadTiming timing;
                const auto before = ktxReader->stats();
                result = createKtxMaterialTexture(device,stagingArena,*ktxReader,decodedKtxScratch,timing,*ktxImages[imageIndex],
                    ktxFirstMips[imageIndex],materialTexture,log,prefetched);
                auto delta = prefetched ? prefetched->stats : ktxReader->stats();
                if (!prefetched) {
                    delta.fileOpens -= before.fileOpens; delta.decodedMips -= before.decodedMips;
                    delta.storedBytes -= before.storedBytes; delta.decodedBytes -= before.decodedBytes;
                    delta.decoderCreations -= before.decoderCreations; delta.openMs -= before.openMs;
                    delta.readMs -= before.readMs; delta.decodeMs -= before.decodeMs;
                }
                auto& stats = uploadStats.ktx;
                stats.fileOpens += delta.fileOpens; stats.decodedMips += delta.decodedMips;
                stats.storedBytes += delta.storedBytes; stats.decodedBytes += delta.decodedBytes;
                stats.decoderCreations += delta.decoderCreations; stats.openMs += delta.openMs;
                stats.readMs += delta.readMs; stats.decodeMs += delta.decodeMs;
                if (prefetched) {
                    uploadStats.prefetch = ktxPrefetch->stats();
                    ktxPrefetch->pop();
                }
                uploadStats.textureBuildMs += timing.totalMs;
                uploadStats.imageCreateMs += timing.imageCreateMs;
                uploadStats.stagingMs += timing.stagingMs;
                uploadStats.stagingCopyMs += timing.copyMs;
                uploadStats.stagingFlushMs += timing.flushMs;
                auto& slowest = uploadStats.slowestTextures;
                slowest.push_back(std::move(timing));
                std::sort(slowest.begin(), slowest.end(), [](const auto& a, const auto& b) { return a.totalMs > b.totalMs; });
                if (slowest.size() > 5) { slowest.resize(5); }
            } else {
                DecodedMaterialTexture decodedTexture;
                if (!decodeSceneTexture(loadedScene,textureIndex,decodedTexture,log) ||
                    (decodedTexture.pixels.empty() && decodedTexture.preparedMips == nullptr)) { return {}; }
                result = createMaterialTexture(device,stagingArena,decodedTexture.pixels.data(),
                    decodedTexture.width,decodedTexture.height,decodedTexture.label,materialTexture,log,decodedTexture.preparedMips);
            }
            if (!result) { ktxPrefetch.reset(); return result; }
            if (materialTexture.texture->allocationSize() > textureBudgetBytes -
                std::min(textureStats.residentAllocationBytes,textureBudgetBytes)) {
                log = "Material texture allocation exceeds budget";
                ktxPrefetch.reset();
                return makeError(Error::OutOfMemory);
            }
            trackTexture(materialTexture);
            const uint32_t materialTextureIndex = static_cast<uint32_t>(materialTextures.size());
            textureIndexMap[textureIndex] = materialTextureIndex;
            if (logicalTexture.imageIndex >= 0 &&
                static_cast<size_t>(logicalTexture.imageIndex) < asyncImageTextureIndexMap.size()) {
                asyncImageTextureIndexMap[static_cast<size_t>(logicalTexture.imageIndex)] =
                    materialTextureIndex;
            }
            pendingTextures.push_back(materialTextureIndex);
            pendingTextureBytes += materialTexture.uploadAllocationSize;
            pendingTextureRegions += materialTexture.mipUploads.size();
            materialTextures.push_back(std::move(materialTexture));
            return {};
        }

        TextureView* fallbackView = materialTextures.front().view.get();
        if (fallbackView == nullptr) {
            return makeError(Error::Failure);
        }
        materialTextureViews.assign(materialTextures.size(),fallbackView);
        for (uint32_t textureIndex = 0; textureIndex < materialTextures.size(); ++textureIndex) {
            if (materialTextures[textureIndex].view == nullptr) {
                return makeError(Error::Failure);
            }
            materialTextureViews[textureIndex] = materialTextures[textureIndex].view.get();
        }
        materialTextureCount = static_cast<uint32_t>(materialTextures.size());
        ktxPrefetch.reset();
        initializeTextureStreaming();
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

        for (const auto index : pendingTextures) {
            transitionTexture(materialTextures[index]);
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
        if (!uploadBatches.empty()) {
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
        resetTextureStreaming();
        waitForTextureUploads();
        bufferUploads.clear();
        stagingArena.clear();
        shadingVertexBuffer.reset();
        fallbackPositionBuffer.reset();
        indexBuffer.reset();
        primitiveBuffer.reset();
        instanceBuffer.reset();
        materialBuffer.reset();
        neuralTextures.clear();
        materialTextures.clear();
        materialTextureViews.clear();
        materialTextureCount = 0;
    }

    void clear()
    {
        ktxPrefetch.reset();
        prefetchedImages.clear();
        textureDecodePending = false;
        decodeWaitBegin.reset();
        resetGpuBuffers();
        ktxImages.clear();
        ktxFirstMips.clear();
        ktxReader.reset();
        decodedKtxScratch = {};
        textureStats = {};
        rtxBuilder.clear();
        drawBounds = scene::Bounds{};
        scenePath.clear();
        prepared = false;
        materialOnly = false;
        uploadStats = {};
        uploadProfileReported = false;
        backpressureBegin.reset();
        pendingTextures.clear();
        pendingTextureBytes = pendingTextureRegions = 0;
        asyncPrepareStage = AsyncPrepareStage::Idle;
        partialUploadResumeStage = AsyncPrepareStage::Idle;
        asyncScene = nullptr;
        asyncScenePath.clear();
        sourceResourceIdentity = 0;
        sourceStructuralRevision = 0;
        sourceGeometryTransformRevision = 0;
        sourceVisibilityRevision = 0;
        sourceMaterialRevision = 0;
        sourceMaterialResourceLayout.clear();
        asyncSourceResourceIdentity = 0;
        asyncSourceStructuralRevision = 0;
        asyncSourceGeometryTransformRevision = 0;
        asyncSourceVisibilityRevision = 0;
        asyncSourceMaterialRevision = 0;
        asyncGpuScene = ScenePathTraceGpuScene{};
        asyncReferencedTextures.clear();
        asyncBufferStep = 0;
    }

    bool valid() const
    {
        return prepared &&
            drawBounds.valid &&
            (materialOnly || (rtxBuilder.valid() && shadingVertexBuffer != nullptr &&
                (device->capabilities().rayTracingPositionFetch || fallbackPositionBuffer != nullptr) &&
                indexBuffer != nullptr && primitiveBuffer != nullptr && instanceBuffer != nullptr)) &&
            materialBuffer != nullptr &&
            !materialTextures.empty() &&
            !materialTextureViews.empty() && materialTextureViews[0] != nullptr;
    }

    bool sourceTopologyMatches(const scene::Scene& sourceScene) const
    {
        return sourceResourceIdentity == sourceScene.resourceIdentity() &&
            sourceStructuralRevision ==
                sourceScene.sceneGraph().structuralRevision() &&
            sourceVisibilityRevision == sourceScene.visibilityRevision() &&
            (sourceMaterialRevision == sourceScene.materialRevision() ||
             sourceMaterialResourceLayout == materialResourceLayout(sourceScene));
    }

    bool textureSettingsMatch(const RenderGraphProperties& properties) const
    {
        return textureStreaming == properties.value("materialTextureStreaming", false) &&
            textureRefineDimension == uint32_t(std::clamp(properties.value("materialTextureRefineDimension",512),1,4096)) &&
            textureColdFrames == uint32_t(std::clamp(properties.value("materialTextureColdFrames",180),2,36000)) &&
            textureMaskMaxDimension == uint32_t(std::clamp(properties.value("materialTextureMaskMaxDimension",0),0,32768)) &&
            textureMaxDimension == uint32_t(std::clamp(properties.value("materialTextureMaxDimension",512),1,32768)) &&
            textureBudgetBytes == uint64_t(std::clamp(properties.value("materialTextureBudgetMiB",2048),1,65536)) * 1024 * 1024;
    }

    void stampSource(const scene::Scene& sourceScene)
    {
        sourceResourceIdentity = sourceScene.resourceIdentity();
        sourceStructuralRevision =
            sourceScene.sceneGraph().structuralRevision();
        sourceGeometryTransformRevision = sourceScene.geometryTransformRevision();
        sourceVisibilityRevision = sourceScene.visibilityRevision();
        sourceMaterialRevision = sourceScene.materialRevision();
        sourceMaterialResourceLayout = materialResourceLayout(sourceScene);
    }

    SceneAccelerationStructureBuilder rtxBuilder;
    bool materialOnly = false;
    Device* device = nullptr;
    Queue* graphicsQueue = nullptr;
    scene::Bounds drawBounds;
    std::filesystem::path scenePath;
    bool prepared = false;
    uint64_t revision = 0;
    uint64_t sourceResourceIdentity = 0;
    uint64_t sourceStructuralRevision = 0;
    uint64_t sourceGeometryTransformRevision = 0;
    uint64_t sourceVisibilityRevision = 0;
    uint64_t sourceMaterialRevision = 0;
    std::vector<std::array<int32_t, 21>> sourceMaterialResourceLayout;
    std::unique_ptr<Buffer> shadingVertexBuffer;
    std::unique_ptr<Buffer> fallbackPositionBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    std::unique_ptr<Buffer> primitiveBuffer;
    std::unique_ptr<Buffer> instanceBuffer;
    std::unique_ptr<Buffer> materialBuffer;
    NeuralTextureResources neuralTextures;
    SceneUploadStagingArena stagingArena;
    std::vector<ScenePathTraceBufferUpload> bufferUploads;
    std::vector<ScenePathTraceMaterialTexture> materialTextures;
    std::vector<uint32_t> pendingTextures;
    uint64_t pendingTextureBytes = 0, pendingTextureRegions = 0;
    std::vector<uint32_t> textureIndexMap;
    std::vector<uint32_t> asyncImageTextureIndexMap;
    std::vector<bool> asyncReferencedTextures;
    size_t asyncTextureCursor = 0;
    std::vector<TextureView*> materialTextureViews;
    std::vector<std::optional<Ktx2TextureInfo>> ktxImages;
    std::vector<uint32_t> ktxFirstMips;
    std::unique_ptr<Ktx2MipReader> ktxReader;
    static constexpr uint64_t kTexturePrefetchBytes = 64ull * 1024 * 1024;
    uint32_t textureLoadWorkers = 4;
    std::unique_ptr<Ktx2TexturePrefetch> ktxPrefetch;
    std::vector<bool> prefetchedImages;
    bool textureDecodePending = false;
    std::optional<SceneResourceLogClock::time_point> decodeWaitBegin;
    std::vector<uint8_t> decodedKtxScratch;
    SceneResourceLogClock::time_point textureLoadBegin{}, lastUploadProgress{};
    std::optional<SceneResourceLogClock::time_point> backpressureBegin;
    SceneResourceLogClock::time_point finalWaitBegin{};
    bool uploadProfileReported = false;
    uint64_t textureBudgetBytes = 2048ull * 1024 * 1024;
    uint32_t textureMaxDimension = 512;
    uint32_t textureMaskMaxDimension = 0;
    bool textureStreaming = false;
    uint32_t textureRefineDimension = 512, textureColdFrames = 180;
    SceneTextureStats textureStats;
    uint32_t materialTextureCount = 0;
    std::deque<std::unique_ptr<UploadBatch>> uploadBatches;
    SceneUploadStats uploadStats;
    AsyncPrepareStage asyncPrepareStage = AsyncPrepareStage::Idle;
    AsyncPrepareStage partialUploadResumeStage = AsyncPrepareStage::Idle;
    const scene::Scene* asyncScene = nullptr;
    std::filesystem::path asyncScenePath;
    uint64_t asyncSourceResourceIdentity = 0;
    uint64_t asyncSourceStructuralRevision = 0;
    uint64_t asyncSourceGeometryTransformRevision = 0;
    uint64_t asyncSourceVisibilityRevision = 0;
    uint64_t asyncSourceMaterialRevision = 0;
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
    if (runtimeScene && runtimeScene->hasStreamGeometry()) {
        Result result = beginPrepareAsync(device, graphicsQueue, properties, *runtimeScene, log);
        bool complete = false;
        scene::SceneLoadProgress progress;
        while (result && !complete) {
            result = pumpPrepareAsync(std::numeric_limits<double>::max(), complete, progress, log);
            if (result && !complete) {
                if (impl_->textureDecodePending) { impl_->ktxPrefetch->wait(); }
                else { std::this_thread::yield(); }
            }
        }
        return result;
    }
    impl_->device = &device;
    impl_->graphicsQueue = &graphicsQueue;
    const std::filesystem::path path = scenePathFromProperties(properties);
    const scene::Scene* boundScene = runtimeSceneForPath(runtimeScene, path);
    if (impl_->valid() && !impl_->materialOnly && impl_->scenePath == path && impl_->textureSettingsMatch(properties) &&
        boundScene != nullptr && impl_->sourceTopologyMatches(*boundScene) &&
        (impl_->sourceGeometryTransformRevision != boundScene->geometryTransformRevision() ||
         impl_->sourceMaterialRevision != boundScene->materialRevision())) {
        return syncRuntimeScene(boundScene, log);
    }
    if (impl_->valid() && !impl_->materialOnly && impl_->scenePath == path && impl_->textureSettingsMatch(properties) &&
        (boundScene == nullptr ||
         (impl_->sourceTopologyMatches(*boundScene) &&
          impl_->sourceGeometryTransformRevision == boundScene->geometryTransformRevision() &&
          impl_->sourceMaterialRevision == boundScene->materialRevision()))) {
        spdlog::info("[SceneResources] Reuse prepared scene='{}'", path.string());
        return {};
    }

    SceneResourceLogScope prepareScope("prepare scene='" + path.string() + "'");
    impl_->clear();

    impl_->textureBudgetBytes = uint64_t(std::clamp(properties.value("materialTextureBudgetMiB",2048),1,65536)) * 1024 * 1024;
    impl_->textureMaxDimension = uint32_t(std::clamp(properties.value("materialTextureMaxDimension",512),1,32768));
    impl_->textureMaskMaxDimension = uint32_t(std::clamp(properties.value("materialTextureMaskMaxDimension",0),0,32768));
    impl_->textureStreaming = properties.value("materialTextureStreaming",false);
    impl_->textureRefineDimension = uint32_t(std::clamp(properties.value("materialTextureRefineDimension",512),1,4096));
    impl_->textureColdFrames = uint32_t(std::clamp(properties.value("materialTextureColdFrames",180),2,36000));
    impl_->textureLoadWorkers = uint32_t(std::clamp(properties.value("materialTextureLoadWorkers",4),1,8));
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
                !device.capabilities().rayTracingPositionFetch,
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
        if (!gpuScene.positions.empty()) {
            result = uploadStorageBuffer(
                device, gpuScene.positions.data(), gpuScene.positions.size() * sizeof(std::array<float, 3>),
                sizeof(std::array<float, 3>), impl_->fallbackPositionBuffer, log,
                "ScenePathTracePass fallback positions", &impl_->bufferUploads, &impl_->stagingArena);
            if (!result) { impl_->clear(); return result; }
        }
        result = uploadStorageBuffer(
            device,
            gpuScene.vertices.data(),
            static_cast<uint64_t>(gpuScene.vertices.size() * sizeof(SceneShadingVertex)),
            sizeof(SceneShadingVertex),
            impl_->shadingVertexBuffer,
            log,
            "ScenePathTracePass shading vertices",
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
        stampTextureFormats(gpuScene.materials,impl_->materialTextures);
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
    std::string& log,
    bool materialsOnly)
{
    const std::filesystem::path path = scenePathFromProperties(properties);
    const scene::Scene* boundScene = runtimeSceneForPath(&runtimeScene, path);
    if (boundScene == nullptr || !boundScene->bounds().valid) {
        log = "Asynchronous scene preparation requires a matching valid runtime scene";
        return makeError(Error::InvalidArgument);
    }
    if (impl_->valid() && impl_->materialOnly == (materialsOnly || boundScene->hasStreamGeometry()) &&
        impl_->scenePath == path && impl_->textureSettingsMatch(properties) && impl_->sourceTopologyMatches(*boundScene)) {
        return syncRuntimeScene(boundScene, log);
    }

    impl_->clear();
    impl_->device = &device;
    impl_->graphicsQueue = &graphicsQueue;
    impl_->asyncScene = boundScene;
    impl_->textureBudgetBytes = uint64_t(std::clamp(properties.value("materialTextureBudgetMiB",2048),1,65536)) * 1024 * 1024;
    impl_->textureMaxDimension = uint32_t(std::clamp(properties.value("materialTextureMaxDimension",512),1,32768));
    impl_->textureMaskMaxDimension = uint32_t(std::clamp(properties.value("materialTextureMaskMaxDimension",0),0,32768));
    impl_->textureStreaming = properties.value("materialTextureStreaming",false);
    impl_->textureRefineDimension = uint32_t(std::clamp(properties.value("materialTextureRefineDimension",512),1,4096));
    impl_->textureColdFrames = uint32_t(std::clamp(properties.value("materialTextureColdFrames",180),2,36000));
    impl_->textureLoadWorkers = uint32_t(std::clamp(properties.value("materialTextureLoadWorkers",4),1,8));
    impl_->materialOnly = materialsOnly || boundScene->hasStreamGeometry();
    impl_->asyncScenePath = path;
    impl_->asyncSourceResourceIdentity = boundScene->resourceIdentity();
    impl_->asyncSourceStructuralRevision =
        boundScene->sceneGraph().structuralRevision();
    impl_->asyncSourceGeometryTransformRevision = boundScene->geometryTransformRevision();
    impl_->asyncSourceVisibilityRevision = boundScene->visibilityRevision();
    impl_->asyncSourceMaterialRevision = boundScene->materialRevision();
    impl_->sourceMaterialResourceLayout = materialResourceLayout(*boundScene);
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
            impl_->asyncSourceVisibilityRevision ||
        impl_->asyncScene->materialRevision() != impl_->asyncSourceMaterialRevision) {
        log = "Scene topology or materials changed during asynchronous resource preparation.";
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
        impl_->retireCompletedTextureUploads();
        if (impl_->backpressureBegin && impl_->uploadBatches.size() < Impl::kMaxUploadBatchesInFlight) {
            impl_->uploadStats.backpressureMs += sceneResourceElapsedMilliseconds(*impl_->backpressureBegin);
            impl_->backpressureBegin.reset();
        }
        switch (impl_->asyncPrepareStage) {
        case Impl::AsyncPrepareStage::MaterialTextures: {
            if (impl_->uploadBatches.size() >= Impl::kMaxUploadBatchesInFlight) {
                if (!impl_->backpressureBegin) { impl_->backpressureBegin = SceneResourceLogClock::now(); }
                return {};
            }
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
            if (impl_->textureDecodePending) { return {}; }
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
            continue;
        }
        case Impl::AsyncPrepareStage::GpuPayload:
            if (!buildGpuScene(
                    *impl_->asyncScene,
                    !impl_->device->capabilities().rayTracingPositionFetch,
                    impl_->textureIndexMap,
                    impl_->neuralTextures.logicalTextureSetIndices(),
                    impl_->asyncGpuScene,
                    log, impl_->materialOnly)) {
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return makeError(Error::Failure);
            }
            impl_->asyncPrepareStage = Impl::AsyncPrepareStage::AccelerationStructure;
            progress.phase = scene::SceneLoadPhase::GpuUpload;
            progress.fraction = 0.82f;
            return {};
        case Impl::AsyncPrepareStage::AccelerationStructure: {
            if (impl_->materialOnly) {
                impl_->asyncBufferStep = 4; // Only material payload; no resident buffers or RTAS.
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Buffers;
                continue;
            }
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
            if (impl_->uploadBatches.size() >= Impl::kMaxUploadBatchesInFlight) {
                return {};
            }
            const void* data = nullptr;
            uint64_t byteSize = 0;
            uint32_t stride = 0;
            std::unique_ptr<Buffer>* destination = nullptr;
            const char* label = nullptr;
            switch (impl_->asyncBufferStep) {
            case 0:
                data = impl_->asyncGpuScene.vertices.data();
                byteSize = impl_->asyncGpuScene.vertices.size() * sizeof(SceneShadingVertex);
                stride = sizeof(SceneShadingVertex);
                destination = &impl_->shadingVertexBuffer;
                label = "ScenePathTracePass shading vertices";
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
                stampTextureFormats(impl_->asyncGpuScene.materials,impl_->materialTextures);
                data = impl_->asyncGpuScene.materials.data();
                byteSize = impl_->asyncGpuScene.materials.size() * sizeof(ScenePathTraceGpuMaterial);
                stride = sizeof(ScenePathTraceGpuMaterial);
                destination = &impl_->materialBuffer;
                label = "ScenePathTracePass materials";
                break;
            case 5:
                if (impl_->asyncGpuScene.positions.empty()) {
                    ++impl_->asyncBufferStep;
                    continue;
                }
                data = impl_->asyncGpuScene.positions.data();
                stride = sizeof(std::array<float, 3>);
                byteSize = impl_->asyncGpuScene.positions.size() * stride;
                destination = &impl_->fallbackPositionBuffer;
                label = "ScenePathTracePass fallback positions";
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
            continue;
        }
        case Impl::AsyncPrepareStage::SubmitPartialUploads:
            if (impl_->uploadBatches.size() >= Impl::kMaxUploadBatchesInFlight) {
                return {};
            }
            result = impl_->submitTextureUploads(
                *impl_->device,
                *impl_->graphicsQueue,
                log);
            if (!result) {
                impl_->ktxPrefetch.reset();
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return result;
            }
            progress.phase = scene::SceneLoadPhase::GpuUpload;
            impl_->asyncPrepareStage = impl_->partialUploadResumeStage;
            impl_->partialUploadResumeStage = Impl::AsyncPrepareStage::Idle;
            continue;
        case Impl::AsyncPrepareStage::SubmitUploads:
            if (impl_->uploadBatches.size() >= Impl::kMaxUploadBatchesInFlight) {
                return {};
            }
            result = impl_->submitTextureUploads(
                *impl_->device,
                *impl_->graphicsQueue,
                log);
            if (!result) {
                impl_->asyncPrepareStage = Impl::AsyncPrepareStage::Failed;
                return result;
            }
            impl_->asyncPrepareStage = Impl::AsyncPrepareStage::WaitForGpu;
            impl_->finalWaitBegin = SceneResourceLogClock::now();
            progress.phase = scene::SceneLoadPhase::AccelerationStructures;
            progress.fraction = 0.94f;
            return {};
        case Impl::AsyncPrepareStage::WaitForGpu:
            progress.phase = scene::SceneLoadPhase::AccelerationStructures;
            progress.fraction = 0.97f;
            {
                bool accelerationStructuresComplete = impl_->materialOnly;
                std::string rtxLog;
                result = impl_->materialOnly ? Result{} : impl_->rtxBuilder.pollBuild(
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
            // All copy/acquire work has completed; keep no large idle staging
            // pages attached to the prepared scene during normal rendering.
            impl_->stagingArena.clear();
            impl_->uploadStats.finalWaitMs += sceneResourceElapsedMilliseconds(impl_->finalWaitBegin);
            impl_->finishUploadProfile();
            spdlog::info("[SceneResources] Uploads completed batches={} bytes={} peakInFlight={}/{}",
                impl_->uploadStats.completedBatches, impl_->uploadStats.submittedBytes,
                impl_->uploadStats.peakInFlightBatches, Impl::kMaxUploadBatchesInFlight);
            impl_->drawBounds = impl_->asyncScene->bounds();
            impl_->scenePath = impl_->asyncScenePath;
            impl_->sourceResourceIdentity = impl_->asyncSourceResourceIdentity;
            impl_->sourceStructuralRevision = impl_->asyncSourceStructuralRevision;
            impl_->sourceGeometryTransformRevision = impl_->asyncSourceGeometryTransformRevision;
            impl_->sourceVisibilityRevision = impl_->asyncSourceVisibilityRevision;
            impl_->sourceMaterialRevision = impl_->asyncSourceMaterialRevision;
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
        log = "Scene resources have no device or graphics queue for a runtime scene update.";
        return makeError(Error::InvalidArgument);
    }
    if (!impl_->sourceTopologyMatches(*boundScene)) {
        Device& device = *impl_->device;
        Queue& graphicsQueue = *impl_->graphicsQueue;
        const RenderGraphProperties properties{
            {"path", impl_->scenePath.string()},
            {"materialTextureMaxDimension", impl_->textureMaxDimension},
            {"materialTextureMaskMaxDimension", impl_->textureMaskMaxDimension},
            {"materialTextureStreaming", impl_->textureStreaming},
            {"materialTextureRefineDimension", impl_->textureRefineDimension},
            {"materialTextureColdFrames", impl_->textureColdFrames},
            {"materialTextureBudgetMiB", impl_->textureBudgetBytes / (1024 * 1024)},
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
    if (impl_->sourceMaterialRevision != boundScene->materialRevision()) {
        std::vector<ScenePathTraceGpuMaterial> materials = buildGpuMaterials(
            *boundScene,
            impl_->textureIndexMap,
            impl_->neuralTextures.logicalTextureSetIndices(),
            log);
        stampTextureFormats(materials,impl_->materialTextures);
        const Result result = uploadStorageBuffer(
            *impl_->device,
            materials.data(),
            static_cast<uint64_t>(materials.size() * sizeof(ScenePathTraceGpuMaterial)),
            sizeof(ScenePathTraceGpuMaterial),
            impl_->materialBuffer,
            log,
            "ScenePathTracePass updated materials");
        if (!result) {
            return result;
        }
        // Do not stamp geometry revisions here: a simultaneous transform edit
        // still needs the instance upload and TLAS refit below.
        impl_->sourceMaterialRevision = boundScene->materialRevision();
        ++impl_->revision;
    }
    // A transform edit on a light/camera must not repack every vertex/index,
    // recalculate per-triangle ray-cone LOD, replace the instance buffer, or
    // submit a synchronous TLAS update when no render instance moved.
    if (impl_->sourceGeometryTransformRevision == boundScene->geometryTransformRevision()) {
        return {};
    }

    if (impl_->materialOnly) {
        impl_->drawBounds = boundScene->bounds();
        impl_->stampSource(*boundScene);
        ++impl_->revision;
        return {};
    }
    ScenePathTraceGpuScene gpuScene;
    if (!buildGpuScene(
            *boundScene,
            false, // Only the instance payload is uploaded during a transform edit.
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
    if (auto* frame = commandBuffer.frameContext()) {
        frame->retain(impl_);
        frame->retain(impl_->textureGeneration);
        if (impl_->texturePublication.valid()) {
            auto result = frame->addDependency(impl_->texturePublication);
            if (!result) { return result; }
        }
    }
    return impl_->uploadMaterialTextures(commandBuffer);
}

Result ScenePathTraceResources::beginTextureStreaming(CommandBuffer& commands, uint64_t frameIndex, Buffer*& feedback, CpuProfileRecorder* profiler)
{
    return impl_->beginTextureStreaming(commands, frameIndex, feedback, profiler);
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

Buffer* ScenePathTraceResources::shadingVertexBuffer() const
{
    return impl_->shadingVertexBuffer.get();
}

Buffer* ScenePathTraceResources::fallbackPositionBuffer() const
{
    return impl_->fallbackPositionBuffer.get();
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

const std::vector<TextureView*>& ScenePathTraceResources::materialTextureViews() const
{
    return impl_->materialTextureViews;
}

const std::vector<uint32_t>& ScenePathTraceResources::materialTextureFirstMips() const
{
    return impl_->ktxFirstMips;
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

SceneUploadStats ScenePathTraceResources::uploadStats() const
{
    return impl_->uploadStats;
}

SceneTextureStats ScenePathTraceResources::textureStats() const { return impl_->textureStats; }
std::span<const uint32_t> ScenePathTraceResources::logicalTextureIndices() const { return impl_->textureIndexMap; }

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
    const bool migrationComplete = !impl_->textureMigration || !impl_->textureMigration->submitted ||
        impl_->textureMigration->frame.completion().isComplete();
    return migrationComplete && impl_->textureUploadsReady() && (accelerationStructureComplete || !result);
}

} // namespace metallic::render
