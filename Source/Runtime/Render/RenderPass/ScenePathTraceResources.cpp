#include "Runtime/Render/RenderPass/ScenePathTraceResources.h"

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
    struct TextureInfo {
        uint32_t textureIndex = kInvalidMaterialTextureIndex;
        uint32_t texCoord = 0;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
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
    std::unique_ptr<Buffer> uploadBuffer;
    uint32_t width = 1;
    uint32_t height = 1;
    uint64_t byteSize = 4;
};

struct ScenePathTraceMaterialTexture {
    std::vector<ScenePathTraceTextureMipUpload> mipUploads;
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
};

struct DecodedEnvironmentTexture {
    std::vector<float> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    std::string label;
};

struct EnvironmentAliasEntry {
    float probability = 1.0f;
    uint32_t aliasIndex = 0;
    float texelProbability = 1.0f;
    uint32_t padding = 0;
};

struct EnvironmentImportanceData {
    std::vector<EnvironmentAliasEntry> aliasTable;
    uint32_t texelCount = 1;
};

static_assert(sizeof(EnvironmentAliasEntry) == 16);

std::string resultMessage(std::string_view label, const Result& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

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

std::filesystem::path environmentPathFromProperties(const RenderGraphProperties& props)
{
    if (!props.contains("environment") || !props["environment"].is_object()) {
        return {};
    }
    const RenderGraphProperties& environment = props["environment"];
    if (!environment.contains("path") || !environment["path"].is_string()) {
        return {};
    }

    std::filesystem::path path = environment["path"].get<std::string>();
    if (path.empty()) {
        return {};
    }
    if (path.is_relative()) {
        path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
    }
    return path;
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
    std::string_view label)
{
    if (data == nullptr || byteSize == 0) {
        log = std::string(label) + " upload data is empty";
        return makeError(Error::InvalidArgument);
    }

    Result result = device.createBuffer(
        BufferDesc{
            .size = byteSize,
            .structureStride = structureStride,
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload,
        },
        outBuffer);
    if (!result || outBuffer == nullptr) {
        log += resultMessage(std::string("createBuffer(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }

    void* mapped = outBuffer->map();
    if (mapped == nullptr) {
        log = std::string(label) + " failed to map upload buffer";
        return makeError(Error::Failure);
    }
    std::memcpy(mapped, data, static_cast<size_t>(byteSize));
    outBuffer->flush(0, byteSize);
    outBuffer->unmap();
    return {};
}

Result createTextureUploadBuffer(
    Device& device,
    const void* data,
    uint64_t byteSize,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log,
    std::string_view label)
{
    if (data == nullptr || byteSize == 0) {
        return makeError(Error::InvalidArgument);
    }

    Result result = device.createBuffer(
        BufferDesc{
            .size = byteSize,
            .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
        },
        outBuffer);
    if (!result || outBuffer == nullptr) {
        log += resultMessage(std::string("createBuffer(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }

    void* mapped = outBuffer->map();
    if (mapped == nullptr) {
        log = std::string(label) + " failed to map upload buffer";
        return makeError(Error::Failure);
    }
    std::memcpy(mapped, data, static_cast<size_t>(byteSize));
    outBuffer->flush(0, byteSize);
    outBuffer->unmap();
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
    const uint8_t* pixels,
    uint32_t width,
    uint32_t height,
    std::string_view label,
    ScenePathTraceMaterialTexture& outTexture,
    std::string& log)
{
    if (pixels == nullptr || width == 0 || height == 0) {
        return makeError(Error::InvalidArgument);
    }

    outTexture = ScenePathTraceMaterialTexture{};
    outTexture.width = width;
    outTexture.height = height;
    outTexture.format = Format::Rgba8Unorm;
    std::vector<DecodedMaterialTexture> mipChain = buildMaterialMipChain(pixels, width, height, label);
    outTexture.mipCount = static_cast<uint32_t>(mipChain.size());
    outTexture.mipUploads.reserve(mipChain.size());
    outTexture.byteSize = 0;
    for (uint32_t mipIndex = 0; mipIndex < mipChain.size(); ++mipIndex) {
        const DecodedMaterialTexture& mip = mipChain[mipIndex];
        ScenePathTraceTextureMipUpload upload;
        upload.width = mip.width;
        upload.height = mip.height;
        upload.byteSize = rgba8ByteSize(mip.width, mip.height);
        outTexture.byteSize += upload.byteSize;

        std::string mipLabel = "ScenePathTracePass texture upload ";
        mipLabel += std::string(label);
        mipLabel += " mip ";
        mipLabel += std::to_string(mipIndex);
        Result result = createTextureUploadBuffer(
            device,
            mip.pixels.data(),
            upload.byteSize,
            upload.uploadBuffer,
            log,
            mipLabel);
        if (!result) {
            return result;
        }
        outTexture.mipUploads.push_back(std::move(upload));
    }

    Result result = device.createTexture(
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

bool decodeEnvironmentTexture(
    const std::filesystem::path& path,
    DecodedEnvironmentTexture& outTexture,
    std::string& log)
{
    outTexture = DecodedEnvironmentTexture{};
    if (path.empty()) {
        return false;
    }

    std::error_code existsError;
    if (!std::filesystem::exists(path, existsError)) {
        appendScenePathTraceWarning(log, "environment map does not exist: " + path.string());
        return false;
    }

    int width = 0;
    int height = 0;
    int channelCount = 0;
    float* pixels = stbi_loadf(path.string().c_str(), &width, &height, &channelCount, 4);
    if (pixels == nullptr || width <= 0 || height <= 0) {
        std::string message = "failed to decode environment map '" + path.string() + "'";
        if (const char* reason = stbi_failure_reason()) {
            message += ": ";
            message += reason;
        }
        appendScenePathTraceWarning(log, message);
        return false;
    }

    const uint64_t componentCount = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ull;
    if (componentCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        stbi_image_free(pixels);
        appendScenePathTraceWarning(log, "decoded environment map is too large");
        return false;
    }

    outTexture.width = static_cast<uint32_t>(width);
    outTexture.height = static_cast<uint32_t>(height);
    outTexture.label = path.filename().string();
    outTexture.pixels.assign(pixels, pixels + static_cast<size_t>(componentCount));
    stbi_image_free(pixels);
    return true;
}

float environmentTexelWeight(const float* rgba, uint32_t y, uint32_t height)
{
    constexpr float kPi = 3.14159265358979323846f;
    const float r = std::isfinite(rgba[0]) ? std::max(rgba[0], 0.0f) : 0.0f;
    const float g = std::isfinite(rgba[1]) ? std::max(rgba[1], 0.0f) : 0.0f;
    const float b = std::isfinite(rgba[2]) ? std::max(rgba[2], 0.0f) : 0.0f;
    const float luminance = r * 0.2126f + g * 0.7152f + b * 0.0722f;
    const float theta = (static_cast<float>(y) + 0.5f) * (kPi / static_cast<float>(std::max(height, 1u)));
    return luminance * std::max(std::sin(theta), 0.0f);
}

EnvironmentImportanceData buildEnvironmentImportanceData(const DecodedEnvironmentTexture& texture)
{
    EnvironmentImportanceData data;
    const uint64_t texelCount64 = static_cast<uint64_t>(texture.width) * static_cast<uint64_t>(texture.height);
    if (texture.pixels.empty() || texture.width == 0 || texture.height == 0 || texelCount64 == 0 ||
        texelCount64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        data.aliasTable = {EnvironmentAliasEntry{}};
        data.texelCount = 1;
        return data;
    }

    data.texelCount = static_cast<uint32_t>(texelCount64);
    data.aliasTable.resize(data.texelCount);
    std::vector<double> weights(data.texelCount, 0.0);
    double totalWeight = 0.0;
    for (uint32_t texelIndex = 0; texelIndex < data.texelCount; ++texelIndex) {
        const uint32_t y = texelIndex / texture.width;
        const float weight = environmentTexelWeight(&texture.pixels[static_cast<size_t>(texelIndex) * 4u], y, texture.height);
        totalWeight += static_cast<double>(weight);
        weights[texelIndex] = static_cast<double>(weight);
    }

    std::vector<double> scaledProbabilities(data.texelCount, 1.0);
    if (totalWeight <= 0.0 || !std::isfinite(totalWeight)) {
        for (uint32_t texelIndex = 0; texelIndex < data.texelCount; ++texelIndex) {
            data.aliasTable[texelIndex].texelProbability = 1.0f / static_cast<float>(data.texelCount);
        }
    } else {
        const double reciprocalTotalWeight = 1.0 / totalWeight;
        for (uint32_t texelIndex = 0; texelIndex < data.texelCount; ++texelIndex) {
            const double texelProbability = weights[texelIndex] * reciprocalTotalWeight;
            data.aliasTable[texelIndex].texelProbability = static_cast<float>(texelProbability);
            scaledProbabilities[texelIndex] = texelProbability * static_cast<double>(data.texelCount);
        }
    }

    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    small.reserve(data.texelCount);
    large.reserve(data.texelCount);
    for (uint32_t texelIndex = 0; texelIndex < data.texelCount; ++texelIndex) {
        if (scaledProbabilities[texelIndex] < 1.0) {
            small.push_back(texelIndex);
        } else {
            large.push_back(texelIndex);
        }
        data.aliasTable[texelIndex].aliasIndex = texelIndex;
    }

    while (!small.empty() && !large.empty()) {
        const uint32_t smallIndex = small.back();
        small.pop_back();
        const uint32_t largeIndex = large.back();

        data.aliasTable[smallIndex].probability =
            static_cast<float>(std::clamp(scaledProbabilities[smallIndex], 0.0, 1.0));
        data.aliasTable[smallIndex].aliasIndex = largeIndex;

        scaledProbabilities[largeIndex] =
            (scaledProbabilities[largeIndex] + scaledProbabilities[smallIndex]) - 1.0;
        if (scaledProbabilities[largeIndex] < 1.0) {
            large.pop_back();
            small.push_back(largeIndex);
        }
    }

    for (uint32_t texelIndex : large) {
        data.aliasTable[texelIndex].probability = 1.0f;
        data.aliasTable[texelIndex].aliasIndex = texelIndex;
    }
    for (uint32_t texelIndex : small) {
        data.aliasTable[texelIndex].probability = 1.0f;
        data.aliasTable[texelIndex].aliasIndex = texelIndex;
    }
    return data;
}

Result createEnvironmentTexture(
    Device& device,
    const float* pixels,
    uint32_t width,
    uint32_t height,
    std::string_view label,
    ScenePathTraceMaterialTexture& outTexture,
    std::string& log)
{
    if (pixels == nullptr || width == 0 || height == 0) {
        return makeError(Error::InvalidArgument);
    }

    outTexture = ScenePathTraceMaterialTexture{};
    outTexture.width = width;
    outTexture.height = height;
    outTexture.format = Format::Rgba32Sfloat;
    outTexture.mipCount = 1;
    outTexture.byteSize = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ull * sizeof(float);
    ScenePathTraceTextureMipUpload upload;
    upload.width = width;
    upload.height = height;
    upload.byteSize = outTexture.byteSize;
    std::string uploadLabel = "ScenePathTracePass environment upload ";
    uploadLabel += std::string(label);
    Result result = createTextureUploadBuffer(
        device,
        pixels,
        upload.byteSize,
        upload.uploadBuffer,
        log,
        uploadLabel);
    if (!result) {
        return result;
    }
    outTexture.mipUploads.push_back(std::move(upload));

    result = device.createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
            .format = outTexture.format,
            .width = width,
            .height = height,
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        },
        outTexture.texture);
    if (!result || outTexture.texture == nullptr) {
        log += resultMessage(std::string("createTexture(ScenePathTracePass environment texture ") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }

    result = device.createTextureView(
        *outTexture.texture,
        TextureViewDesc{
            .format = outTexture.format,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        },
        outTexture.view);
    if (!result || outTexture.view == nullptr) {
        log += resultMessage(std::string("createTextureView(ScenePathTracePass environment texture ") + std::string(label) + ")", result);
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
    const std::vector<uint32_t>& textureIndexMap,
    std::string& log,
    std::string_view textureLabel)
{
    ScenePathTraceGpuMaterial::TextureInfo gpuTextureInfo;
    gpuTextureInfo.textureIndex = materialTextureIndex(textureInfo.textureIndex, textureIndexMap);
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
    const std::vector<uint32_t>& textureIndexMap,
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
    gpuMaterial.baseColorTexture = makeGpuTextureInfo(material.baseColorTexture, textureIndexMap, log, "baseColorTexture");
    gpuMaterial.metallicRoughnessTexture = makeGpuTextureInfo(
        material.metallicRoughnessTexture,
        textureIndexMap,
        log,
        "metallicRoughnessTexture");
    gpuMaterial.normalTexture = makeGpuTextureInfo(material.normalTexture, textureIndexMap, log, "normalTexture");
    gpuMaterial.occlusionTexture = makeGpuTextureInfo(material.occlusionTexture, textureIndexMap, log, "occlusionTexture");
    gpuMaterial.emissiveTexture = makeGpuTextureInfo(material.emissiveTexture, textureIndexMap, log, "emissiveTexture");
    gpuMaterial.transmissionTexture = makeGpuTextureInfo(
        material.transmissionTexture,
        textureIndexMap,
        log,
        "transmissionTexture");
    gpuMaterial.thicknessTexture = makeGpuTextureInfo(material.thicknessTexture, textureIndexMap, log, "thicknessTexture");
    gpuMaterial.diffuseTransmissionTexture = makeGpuTextureInfo(
        material.diffuseTransmissionTexture,
        textureIndexMap,
        log,
        "diffuseTransmissionTexture");
    gpuMaterial.diffuseTransmissionColorTexture = makeGpuTextureInfo(
        material.diffuseTransmissionColorTexture,
        textureIndexMap,
        log,
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
    ScenePathTraceGpuScene& outScene,
    std::string& log)
{
    outScene = ScenePathTraceGpuScene{};
    outScene.materials.reserve(std::max<size_t>(loadedScene.materials().size(), 1));
    if (loadedScene.materials().empty()) {
        outScene.materials.push_back(ScenePathTraceGpuMaterial{});
    } else {
        for (const scene::RenderMaterial& material : loadedScene.materials()) {
            outScene.materials.push_back(makeMaterial(material, textureIndexMap, log));
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
        outScene.instances.empty() ||
        outScene.materials.empty()) {
        log = "ScenePathTracePass found no visible triangle geometry for path tracing";
        return false;
    }
    return true;
}

} // namespace

struct ScenePathTraceResources::Impl {
    Result buildMaterialTextures(
        Device& device,
        const scene::Scene& loadedScene,
        std::vector<uint32_t>& outTextureIndexMap,
        std::string& log)
    {
        materialTextures.clear();
        materialTextureViews.fill(nullptr);
        materialTextureCount = 0;
        environmentTexture = ScenePathTraceMaterialTexture{};
        environmentMapAvailable = false;
        outTextureIndexMap.assign(loadedScene.textures().size(), kInvalidMaterialTextureIndex);

        const uint8_t fallbackPixels[4] = {255, 255, 255, 255};
        ScenePathTraceMaterialTexture fallbackTexture;
        const auto fallbackBegin = SceneResourceLogClock::now();
        Result result = createMaterialTexture(
            device,
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
        for (uint32_t textureIndex = 0; textureIndex < loadedScene.textures().size(); ++textureIndex) {
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
            if (decodedTexture.pixels.empty()) {
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
            decodedTextureBytes += decodedTexture.pixels.size();

            ScenePathTraceMaterialTexture materialTexture;
            const auto createBegin = SceneResourceLogClock::now();
            result = createMaterialTexture(
                device,
                decodedTexture.pixels.data(),
                decodedTexture.width,
                decodedTexture.height,
                decodedTexture.label,
                materialTexture,
                log);
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

    Result buildEnvironmentTexture(
        Device& device,
        const std::filesystem::path& path,
        std::string& log)
    {
        environmentTexture = ScenePathTraceMaterialTexture{};
        environmentImportanceBuffer.reset();
        environmentImportanceTexelCount = 1;
        environmentMapAvailable = false;

        if (!path.empty()) {
            DecodedEnvironmentTexture decodedEnvironment;
            const auto decodeBegin = SceneResourceLogClock::now();
            if (decodeEnvironmentTexture(path, decodedEnvironment, log) && !decodedEnvironment.pixels.empty()) {
                spdlog::info(
                    "[SceneResources] Environment decoded '{}' {}x{} texels={} in {:.2f} ms",
                    decodedEnvironment.label,
                    decodedEnvironment.width,
                    decodedEnvironment.height,
                    decodedEnvironment.pixels.size() / 4u,
                    sceneResourceElapsedMilliseconds(decodeBegin));
                const auto textureBegin = SceneResourceLogClock::now();
                Result result = createEnvironmentTexture(
                    device,
                    decodedEnvironment.pixels.data(),
                    decodedEnvironment.width,
                    decodedEnvironment.height,
                    decodedEnvironment.label,
                    environmentTexture,
                    log);
                if (!result) {
                    return result;
                }
                spdlog::info(
                    "[SceneResources] Environment GPU texture resources uploadBytes={} in {:.2f} ms",
                    environmentTexture.byteSize,
                    sceneResourceElapsedMilliseconds(textureBegin));
                const auto importanceBegin = SceneResourceLogClock::now();
                EnvironmentImportanceData importanceData = buildEnvironmentImportanceData(decodedEnvironment);
                spdlog::info(
                    "[SceneResources] Environment importance table built texels={} in {:.2f} ms",
                    importanceData.texelCount,
                    sceneResourceElapsedMilliseconds(importanceBegin));
                const auto uploadBegin = SceneResourceLogClock::now();
                result = uploadStorageBuffer(
                    device,
                    importanceData.aliasTable.data(),
                    static_cast<uint64_t>(importanceData.aliasTable.size() * sizeof(EnvironmentAliasEntry)),
                    sizeof(EnvironmentAliasEntry),
                    environmentImportanceBuffer,
                    log,
                    "ScenePathTracePass environment importance alias table");
                if (!result) {
                    return result;
                }
                environmentImportanceTexelCount = importanceData.texelCount;
                environmentMapAvailable = true;
                spdlog::info(
                    "[SceneResources] Environment importance upload bytes={} in {:.2f} ms",
                    importanceData.aliasTable.size() * sizeof(EnvironmentAliasEntry),
                    sceneResourceElapsedMilliseconds(uploadBegin));
                return {};
            }
            spdlog::info(
                "[SceneResources] Environment decode skipped/failed '{}' in {:.2f} ms",
                path.string(),
                sceneResourceElapsedMilliseconds(decodeBegin));
        }

        const float fallbackPixels[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        const auto fallbackBegin = SceneResourceLogClock::now();
        Result result = createEnvironmentTexture(
            device,
            fallbackPixels,
            1,
            1,
            "environment fallback",
            environmentTexture,
            log);
        if (!result) {
            return result;
        }
        spdlog::info(
            "[SceneResources] Environment fallback texture prepared in {:.2f} ms",
            sceneResourceElapsedMilliseconds(fallbackBegin));

        const EnvironmentAliasEntry fallbackAliasTable[1] = {EnvironmentAliasEntry{}};
        const auto uploadBegin = SceneResourceLogClock::now();
        result = uploadStorageBuffer(
            device,
            fallbackAliasTable,
            sizeof(fallbackAliasTable),
            sizeof(EnvironmentAliasEntry),
            environmentImportanceBuffer,
            log,
            "ScenePathTracePass environment fallback importance alias table");
        spdlog::info(
            "[SceneResources] Environment fallback importance upload in {:.2f} ms",
            sceneResourceElapsedMilliseconds(uploadBegin));
        return result;
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
            if (upload.uploadBuffer == nullptr || upload.width == 0 || upload.height == 0) {
                return makeError(Error::InvalidArgument);
            }

            commandBuffer.copyBufferToTexture(BufferTextureCopyDesc{
                .buffer = upload.uploadBuffer.get(),
                .texture = texture.texture.get(),
                .width = upload.width,
                .height = upload.height,
                .depth = 1,
                .mipLevel = mipIndex,
                .baseLayer = 0,
            });
        }

        TextureBarrierDesc toShaderRead{
            .texture = texture.texture.get(),
            .before = texture.state,
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
        texture.uploaded = true;
        return {};
    }

    Result uploadMaterialTextures(CommandBuffer& commandBuffer)
    {
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

    Result uploadEnvironmentTexture(CommandBuffer& commandBuffer)
    {
        return uploadTexture(commandBuffer, environmentTexture);
    }

    void resetGpuBuffers()
    {
        vertexBuffer.reset();
        indexBuffer.reset();
        primitiveBuffer.reset();
        instanceBuffer.reset();
        materialBuffer.reset();
        materialTextures.clear();
        materialTextureViews.fill(nullptr);
        materialTextureCount = 0;
        environmentTexture = ScenePathTraceMaterialTexture{};
        environmentImportanceBuffer.reset();
        environmentImportanceTexelCount = 1;
        environmentMapAvailable = false;
    }

    void clear()
    {
        resetGpuBuffers();
        rtxBuilder.clear();
        drawBounds = scene::Bounds{};
        scenePath.clear();
        environmentPath.clear();
        prepared = false;
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
            materialTextureViews[0] != nullptr &&
            environmentTexture.view != nullptr &&
            environmentImportanceBuffer != nullptr;
    }

    SceneRtxBuilder rtxBuilder;
    scene::Bounds drawBounds;
    std::filesystem::path scenePath;
    std::filesystem::path environmentPath;
    bool prepared = false;
    uint64_t revision = 0;
    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    std::unique_ptr<Buffer> primitiveBuffer;
    std::unique_ptr<Buffer> instanceBuffer;
    std::unique_ptr<Buffer> materialBuffer;
    std::vector<ScenePathTraceMaterialTexture> materialTextures;
    std::array<TextureView*, kScenePathTraceMaxMaterialTextures> materialTextureViews{};
    uint32_t materialTextureCount = 0;
    ScenePathTraceMaterialTexture environmentTexture;
    std::unique_ptr<Buffer> environmentImportanceBuffer;
    uint32_t environmentImportanceTexelCount = 1;
    bool environmentMapAvailable = false;
};
ScenePathTraceResources::ScenePathTraceResources() :
    impl_(std::make_unique<Impl>())
{
}

ScenePathTraceResources::~ScenePathTraceResources() = default;

ScenePathTraceResources::ScenePathTraceResources(ScenePathTraceResources&&) noexcept = default;

ScenePathTraceResources& ScenePathTraceResources::operator=(ScenePathTraceResources&&) noexcept = default;

Result ScenePathTraceResources::prepare(
    Device& device,
    Queue& graphicsQueue,
    const RenderGraphProperties& properties,
    std::string& log)
{
    const std::filesystem::path path = scenePathFromProperties(properties);
    const std::filesystem::path environmentPath = environmentPathFromProperties(properties);
    if (impl_->valid() && impl_->scenePath == path && impl_->environmentPath == environmentPath) {
        spdlog::info(
            "[SceneResources] Reuse prepared scene='{}' environment='{}'",
            path.string(),
            environmentPath.string());
        return {};
    }

    SceneResourceLogScope prepareScope(
        "prepare scene='" + path.string() + "' environment='" + environmentPath.string() + "'");
    impl_->clear();

    scene::Scene loadedScene;
    {
        SceneResourceLogScope scope("load scene for render pass resources");
        if (!loadedScene.load(path)) {
            log = "ScenePathTracePass failed to load glTF: " + loadedScene.lastLoadResult().error;
            return makeError(Error::Failure);
        }
    }
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
    {
        SceneResourceLogScope scope("build environment texture");
        result = impl_->buildEnvironmentTexture(device, environmentPath, log);
    }
    if (!result) {
        impl_->clear();
        return result;
    }

    ScenePathTraceGpuScene gpuScene;
    {
        SceneResourceLogScope scope("build GPU scene payload");
        if (!buildGpuScene(loadedScene, textureIndexMap, gpuScene, log)) {
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
        SceneResourceLogScope scope("build RTX acceleration structures for render pass");
        result = impl_->rtxBuilder.build(device, graphicsQueue, loadedScene, rtxLog);
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
            "ScenePathTracePass vertices");
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
            "ScenePathTracePass indices");
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
            "ScenePathTracePass primitives");
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
            "ScenePathTracePass instances");
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
            "ScenePathTracePass materials");
        if (!result) {
            impl_->clear();
            return result;
        }
    }

    impl_->drawBounds = loadedScene.bounds();
    impl_->scenePath = path;
    impl_->environmentPath = environmentPath;
    impl_->prepared = true;
    ++impl_->revision;
    spdlog::info(
        "[SceneResources] Prepared scene resources revision={} materialTextures={} environmentAvailable={} environmentTexels={}",
        impl_->revision,
        impl_->materialTextureCount,
        impl_->environmentMapAvailable,
        impl_->environmentImportanceTexelCount);
    return {};
}

Result ScenePathTraceResources::uploadMaterialTextures(CommandBuffer& commandBuffer)
{
    return impl_->uploadMaterialTextures(commandBuffer);
}

Result ScenePathTraceResources::uploadEnvironmentTexture(CommandBuffer& commandBuffer)
{
    return impl_->uploadEnvironmentTexture(commandBuffer);
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

SceneRtxBuilder& ScenePathTraceResources::accelerationStructure()
{
    return impl_->rtxBuilder;
}

const SceneRtxBuilder& ScenePathTraceResources::accelerationStructure() const
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

TextureView* ScenePathTraceResources::environmentTextureView() const
{
    return impl_->environmentTexture.view.get();
}

Buffer* ScenePathTraceResources::environmentImportanceBuffer() const
{
    return impl_->environmentImportanceBuffer.get();
}

uint32_t ScenePathTraceResources::environmentImportanceTexelCount() const
{
    return impl_->environmentImportanceTexelCount;
}

bool ScenePathTraceResources::environmentMapAvailable() const
{
    return impl_->environmentMapAvailable;
}

} // namespace metallic::render
