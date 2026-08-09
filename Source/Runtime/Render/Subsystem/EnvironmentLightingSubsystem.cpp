#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

std::filesystem::path resolvedEnvironmentPath(const std::filesystem::path& path)
{
    if (path.empty() || path.is_absolute()) {
        return path;
    }
    return std::filesystem::path(PROJECT_SOURCE_DIR) / path;
}

float environmentTexelWeight(const float* rgba, uint32_t y, uint32_t height)
{
    constexpr float kPi = 3.14159265358979323846f;
    const float r = std::isfinite(rgba[0]) ? std::max(rgba[0], 0.0f) : 0.0f;
    const float g = std::isfinite(rgba[1]) ? std::max(rgba[1], 0.0f) : 0.0f;
    const float b = std::isfinite(rgba[2]) ? std::max(rgba[2], 0.0f) : 0.0f;
    const float luminance = r * 0.2126f + g * 0.7152f + b * 0.0722f;
    const float theta = (static_cast<float>(y) + 0.5f) *
        (kPi / static_cast<float>(std::max(height, 1u)));
    return luminance * std::max(std::sin(theta), 0.0f);
}

std::vector<EnvironmentAliasEntry> buildAliasTable(
    std::span<const float> pixels,
    uint32_t width,
    uint32_t height)
{
    const uint64_t texelCount64 = static_cast<uint64_t>(width) * height;
    if (pixels.size() < texelCount64 * 4ull ||
        texelCount64 == 0 ||
        texelCount64 > std::numeric_limits<uint32_t>::max()) {
        return {EnvironmentAliasEntry{}};
    }

    const uint32_t texelCount = static_cast<uint32_t>(texelCount64);
    std::vector<EnvironmentAliasEntry> table(texelCount);
    std::vector<double> weights(texelCount, 0.0);
    double totalWeight = 0.0;
    for (uint32_t index = 0; index < texelCount; ++index) {
        const uint32_t y = index / width;
        const float weight = environmentTexelWeight(&pixels[static_cast<size_t>(index) * 4u], y, height);
        weights[index] = weight;
        totalWeight += weight;
    }

    std::vector<double> scaled(texelCount, 1.0);
    if (totalWeight > 0.0 && std::isfinite(totalWeight)) {
        for (uint32_t index = 0; index < texelCount; ++index) {
            const double probability = weights[index] / totalWeight;
            table[index].texelProbability = static_cast<float>(probability);
            scaled[index] = probability * texelCount;
        }
    } else {
        for (EnvironmentAliasEntry& entry : table) {
            entry.texelProbability = 1.0f / static_cast<float>(texelCount);
        }
    }

    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    small.reserve(texelCount);
    large.reserve(texelCount);
    for (uint32_t index = 0; index < texelCount; ++index) {
        table[index].aliasIndex = index;
        (scaled[index] < 1.0 ? small : large).push_back(index);
    }
    while (!small.empty() && !large.empty()) {
        const uint32_t smallIndex = small.back();
        small.pop_back();
        const uint32_t largeIndex = large.back();
        table[smallIndex].probability = static_cast<float>(std::clamp(scaled[smallIndex], 0.0, 1.0));
        table[smallIndex].aliasIndex = largeIndex;
        scaled[largeIndex] = scaled[largeIndex] + scaled[smallIndex] - 1.0;
        if (scaled[largeIndex] < 1.0) {
            large.pop_back();
            small.push_back(largeIndex);
        }
    }
    for (uint32_t index : large) {
        table[index].probability = 1.0f;
        table[index].aliasIndex = index;
    }
    for (uint32_t index : small) {
        table[index].probability = 1.0f;
        table[index].aliasIndex = index;
    }
    return table;
}

EnvironmentSphericalHarmonics computeEnvironmentSH(
    std::span<const float> pixels,
    uint32_t width,
    uint32_t height)
{
    EnvironmentSphericalHarmonics coefficients{};
    if (pixels.size() < static_cast<size_t>(width) * height * 4u || width == 0 || height == 0) {
        return coefficients;
    }

    constexpr uint32_t kMaxSampleWidth = 256;
    constexpr uint32_t kMaxSampleHeight = 128;
    constexpr float kPi = 3.14159265358979323846f;
    const uint32_t sampleWidth = std::min(width, kMaxSampleWidth);
    const uint32_t sampleHeight = std::min(height, kMaxSampleHeight);
    const float deltaPhi = 2.0f * kPi / static_cast<float>(sampleWidth);
    const float deltaTheta = kPi / static_cast<float>(sampleHeight);

    for (uint32_t sampleY = 0; sampleY < sampleHeight; ++sampleY) {
        const float v = (static_cast<float>(sampleY) + 0.5f) / sampleHeight;
        const float theta = v * kPi;
        const float sineTheta = std::sin(theta);
        const float directionY = std::cos(theta);
        const uint32_t sourceY = std::min(static_cast<uint32_t>(v * height), height - 1u);
        const float solidAngle = sineTheta * deltaPhi * deltaTheta;
        for (uint32_t sampleX = 0; sampleX < sampleWidth; ++sampleX) {
            const float u = (static_cast<float>(sampleX) + 0.5f) / sampleWidth;
            const float phi = (u - 0.5f) * 2.0f * kPi;
            const float directionX = std::cos(phi) * sineTheta;
            const float directionZ = std::sin(phi) * sineTheta;
            const uint32_t sourceX = std::min(static_cast<uint32_t>(u * width), width - 1u);
            const size_t pixelIndex = (static_cast<size_t>(sourceY) * width + sourceX) * 4u;
            const std::array<float, 3> radiance{
                std::max(pixels[pixelIndex], 0.0f),
                std::max(pixels[pixelIndex + 1u], 0.0f),
                std::max(pixels[pixelIndex + 2u], 0.0f),
            };
            const std::array<float, 9> basis{
                0.282095f,
                0.488603f * directionY,
                0.488603f * directionZ,
                0.488603f * directionX,
                1.092548f * directionX * directionY,
                1.092548f * directionY * directionZ,
                0.315392f * (3.0f * directionZ * directionZ - 1.0f),
                1.092548f * directionX * directionZ,
                0.546274f * (directionX * directionX - directionY * directionY),
            };
            for (size_t coefficient = 0; coefficient < coefficients.size(); ++coefficient) {
                for (size_t channel = 0; channel < radiance.size(); ++channel) {
                    coefficients[coefficient][channel] +=
                        radiance[channel] * basis[coefficient] * solidAngle;
                }
            }
        }
    }
    return coefficients;
}

struct EnvironmentUploadResources {
    std::unique_ptr<Buffer> radiance;
    std::unique_ptr<Buffer> importance;
};

} // namespace

struct EnvironmentLightingSubsystem::DecodedEnvironment {
    uint64_t generation = 0;
    std::vector<float> pixels;
    std::vector<EnvironmentAliasEntry> importance;
    EnvironmentSphericalHarmonics sphericalHarmonics{};
    uint32_t width = 1;
    uint32_t height = 1;
    bool mapAvailable = false;
    bool placeholder = false;
    std::string error;
};

struct EnvironmentLightingSubsystem::DecodeJob {
    std::future<DecodedEnvironment> future;
};

struct EnvironmentLightingSubsystem::Resources {
    std::unique_ptr<Texture> radiance;
    std::unique_ptr<TextureView> radianceView;
    std::unique_ptr<Buffer> importanceBuffer;
    ImportancePdfTexture pdf;
    std::vector<EnvironmentAliasEntry> cpuImportance;
    EnvironmentSphericalHarmonics sphericalHarmonics{};
    uint32_t width = 1;
    uint32_t height = 1;
    bool mapAvailable = false;
};

EnvironmentLightingSubsystem::EnvironmentLightingSubsystem() = default;
EnvironmentLightingSubsystem::~EnvironmentLightingSubsystem() = default;

Result EnvironmentLightingSubsystem::initialize(
    const RenderSubsystemInitContext& context,
    std::string& log)
{
    device_ = &context.device;
    if (const Desc* desc = context.host.configuration<EnvironmentLightingSubsystem>()) {
        desc_ = *desc;
    }
    desc_.maxDecodeJobs = std::max(desc_.maxDecodeJobs, 1u);
    const DeviceCapabilities& capabilities = context.device.capabilities();
    if (capabilities.rayTracingAccelerationStructure && capabilities.rayQuery) {
        return pdfCompute_.initialize(context.device, log);
    }
    return {};
}

void EnvironmentLightingSubsystem::onWorldChanged(RenderWorld* world)
{
    world_ = world;
    if (world_ == nullptr) {
        requestEnvironment(EnvironmentSettings{}, 0);
        return;
    }
    requestEnvironment(world_->environment(), world_->environmentRevision());
}

void EnvironmentLightingSubsystem::requestEnvironment(
    const EnvironmentSettings& settings,
    uint64_t settingsRevision)
{
    if (requestInitialized_ &&
        requestedSettings_ == settings &&
        requestedSettingsRevision_ == settingsRevision) {
        return;
    }
    if (requestInitialized_ &&
        resolvedEnvironmentPath(requestedSettings_.path) == resolvedEnvironmentPath(settings.path)) {
        requestedSettings_ = settings;
        requestedSettingsRevision_ = settingsRevision;
        refreshSnapshot();
        return;
    }
    requestInitialized_ = true;
    requestedSettings_ = settings;
    requestedSettingsRevision_ = settingsRevision;
    const uint64_t generation = ++requestedGeneration_;
    snapshot_.settings = settings;
    snapshot_.settingsRevision = settingsRevision;
    snapshot_.status = EnvironmentLightingStatus::Loading;
    snapshot_.error.clear();
    pendingDecodePath_.clear();
    pendingDecodeGeneration_ = 0;

    const std::filesystem::path path = resolvedEnvironmentPath(settings.path);
    if (path.empty()) {
        auto decoded = std::make_unique<DecodedEnvironment>();
        decoded->generation = generation;
        decoded->pixels = {0.0f, 0.0f, 0.0f, 1.0f};
        decoded->importance = {EnvironmentAliasEntry{}};
        readyDecode_ = std::move(decoded);
        return;
    }

    if (resources_ == nullptr) {
        auto placeholder = std::make_unique<DecodedEnvironment>();
        placeholder->generation = generation;
        placeholder->pixels = {0.0f, 0.0f, 0.0f, 1.0f};
        placeholder->importance = {EnvironmentAliasEntry{}};
        placeholder->placeholder = true;
        readyDecode_ = std::move(placeholder);
    }

    pendingDecodePath_ = path;
    pendingDecodeGeneration_ = generation;
    if (decodeJobs_.size() < desc_.maxDecodeJobs) {
        startDecodeJob(pendingDecodePath_, pendingDecodeGeneration_);
    }
}

void EnvironmentLightingSubsystem::startDecodeJob(
    const std::filesystem::path& path,
    uint64_t generation)
{
    if (path.empty() || generation == 0 || decodeJobs_.size() >= desc_.maxDecodeJobs) {
        return;
    }
    const std::filesystem::path pathToDecode = path;
    pendingDecodePath_.clear();
    pendingDecodeGeneration_ = 0;
    DecodeJob job;
    job.future = std::async(std::launch::async, [pathToDecode, generation]() {
        DecodedEnvironment decoded;
        decoded.generation = generation;
        int width = 0;
        int height = 0;
        int channels = 0;
        float* pixels = stbi_loadf(pathToDecode.string().c_str(), &width, &height, &channels, 4);
        if (pixels == nullptr || width <= 0 || height <= 0) {
            decoded.error = "Failed to decode environment map '" + pathToDecode.string() + "'";
            if (const char* reason = stbi_failure_reason()) {
                decoded.error += ": ";
                decoded.error += reason;
            }
            if (pixels != nullptr) {
                stbi_image_free(pixels);
            }
            return decoded;
        }
        const uint64_t componentCount = static_cast<uint64_t>(width) * height * 4ull;
        if (componentCount > std::numeric_limits<size_t>::max()) {
            stbi_image_free(pixels);
            decoded.error = "Decoded environment map is too large: " + pathToDecode.string();
            return decoded;
        }
        decoded.width = static_cast<uint32_t>(width);
        decoded.height = static_cast<uint32_t>(height);
        decoded.pixels.assign(pixels, pixels + static_cast<size_t>(componentCount));
        stbi_image_free(pixels);
        decoded.importance = buildAliasTable(decoded.pixels, decoded.width, decoded.height);
        decoded.sphericalHarmonics = computeEnvironmentSH(decoded.pixels, decoded.width, decoded.height);
        decoded.mapAvailable = true;
        return decoded;
    });
    decodeJobs_.push_back(std::move(job));
}

Result EnvironmentLightingSubsystem::beginFrame(
    const RenderSubsystemFrameContext&,
    RenderChangeBits& changes,
    std::string&)
{
    if (world_ != nullptr) {
        requestEnvironment(world_->environment(), world_->environmentRevision());
    }
    pollDecodeJobs(changes);
    refreshSnapshot();
    return {};
}

void EnvironmentLightingSubsystem::pollDecodeJobs(RenderChangeBits& changes)
{
    for (size_t index = 0; index < decodeJobs_.size();) {
        DecodeJob& job = decodeJobs_[index];
        if (job.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++index;
            continue;
        }
        DecodedEnvironment decoded = job.future.get();
        decodeJobs_.erase(decodeJobs_.begin() + static_cast<std::ptrdiff_t>(index));
        ++decodeCount_;
        if (decoded.generation != requestedGeneration_) {
            continue;
        }
        if (!decoded.error.empty()) {
            snapshot_.status = EnvironmentLightingStatus::Degraded;
            snapshot_.error = decoded.error;
            if (resources_ == nullptr) {
                decoded.pixels = {0.0f, 0.0f, 0.0f, 1.0f};
                decoded.importance = {EnvironmentAliasEntry{}};
                decoded.width = 1;
                decoded.height = 1;
                decoded.mapAvailable = false;
                readyDecode_ = std::make_unique<DecodedEnvironment>(std::move(decoded));
            }
            continue;
        }
        readyDecode_ = std::make_unique<DecodedEnvironment>(std::move(decoded));
        changes |= RenderChangeBits::Lighting | RenderChangeBits::InvalidateTemporalHistory;
    }
    if (pendingDecodeGeneration_ != 0 &&
        pendingDecodeGeneration_ != requestedGeneration_) {
        pendingDecodePath_.clear();
        pendingDecodeGeneration_ = 0;
    }
    if (pendingDecodeGeneration_ != 0 && decodeJobs_.size() < desc_.maxDecodeJobs) {
        startDecodeJob(pendingDecodePath_, pendingDecodeGeneration_);
    }
}

Result EnvironmentLightingSubsystem::recordPreGraph(
    const RenderSubsystemFrameContext& context,
    std::string& log)
{
    if (context.commandBuffer == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (readyDecode_ == nullptr) {
        return {};
    }
    return publishDecoded(context, std::move(*readyDecode_), log);
}

Result EnvironmentLightingSubsystem::publishDecoded(
    const RenderSubsystemFrameContext& context,
    DecodedEnvironment decoded,
    std::string& log)
{
    readyDecode_.reset();
    if (decoded.generation != requestedGeneration_ || device_ == nullptr) {
        return {};
    }
    if (decoded.pixels.empty()) {
        return makeError(Error::InvalidArgument);
    }

    auto next = std::make_shared<Resources>();
    next->width = std::max(decoded.width, 1u);
    next->height = std::max(decoded.height, 1u);
    next->mapAvailable = decoded.mapAvailable;
    next->cpuImportance = std::move(decoded.importance);
    if (next->cpuImportance.empty()) {
        next->cpuImportance = {EnvironmentAliasEntry{}};
    }
    next->sphericalHarmonics = decoded.sphericalHarmonics;

    Result result = device_->createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
            .format = Format::Rgba32Sfloat,
            .width = next->width,
            .height = next->height,
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
        },
        next->radiance);
    if (!result || next->radiance == nullptr) {
        log = "EnvironmentLightingSubsystem createTexture returned " + std::string(resultToString(result));
        return result ? makeError(Error::Failure) : result;
    }
    result = device_->createTextureView(
        *next->radiance,
        TextureViewDesc{
            .format = Format::Rgba32Sfloat,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        },
        next->radianceView);
    if (!result || next->radianceView == nullptr) {
        log = "EnvironmentLightingSubsystem createTextureView returned " + std::string(resultToString(result));
        return result ? makeError(Error::Failure) : result;
    }
    const uint64_t importanceBytes = next->cpuImportance.size() * sizeof(EnvironmentAliasEntry);
    result = device_->createBuffer(
        BufferDesc{
            .size = importanceBytes,
            .structureStride = sizeof(EnvironmentAliasEntry),
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
        },
        next->importanceBuffer);
    if (!result || next->importanceBuffer == nullptr) {
        log = "EnvironmentLightingSubsystem createBuffer returned " + std::string(resultToString(result));
        return result ? makeError(Error::Failure) : result;
    }
    if (pdfCompute_.valid()) {
        result = next->pdf.initialize(
            *device_,
            next->width,
            next->height,
            "EnvironmentLightingSubsystem PDF",
            log);
        if (!result) {
            return result;
        }
    }

    const uint64_t radianceBytes = decoded.pixels.size() * sizeof(float);
    auto staging = std::make_shared<EnvironmentUploadResources>();
    result = device_->createBuffer(
        BufferDesc{
            .size = radianceBytes,
            .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics,
        },
        staging->radiance);
    if (!result || staging->radiance == nullptr) {
        log = "EnvironmentLightingSubsystem failed to create the radiance staging buffer";
        return result ? makeError(Error::Failure) : result;
    }
    result = device_->createBuffer(
        BufferDesc{
            .size = importanceBytes,
            .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics,
        },
        staging->importance);
    if (!result || staging->importance == nullptr) {
        log = "EnvironmentLightingSubsystem failed to create the importance staging buffer";
        return result ? makeError(Error::Failure) : result;
    }
    void* mappedRadiance = staging->radiance->map();
    void* mappedImportance = staging->importance->map();
    if (mappedRadiance == nullptr || mappedImportance == nullptr) {
        if (mappedRadiance != nullptr) {
            staging->radiance->unmap();
        }
        if (mappedImportance != nullptr) {
            staging->importance->unmap();
        }
        log = "EnvironmentLightingSubsystem failed to map its staging buffers";
        return makeError(Error::Failure);
    }
    std::memcpy(mappedRadiance, decoded.pixels.data(), static_cast<size_t>(radianceBytes));
    std::memcpy(mappedImportance, next->cpuImportance.data(), static_cast<size_t>(importanceBytes));
    staging->radiance->flush(0, radianceBytes);
    staging->importance->flush(0, importanceBytes);
    staging->radiance->unmap();
    staging->importance->unmap();

    TextureBarrierDesc textureToTransfer{
        .texture = next->radiance.get(),
        .before = ResourceState::Undefined,
        .after = ResourceState::TransferDestination,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    BufferBarrierDesc bufferToTransfer{
        .buffer = next->importanceBuffer.get(),
        .before = ResourceState::Undefined,
        .after = ResourceState::TransferDestination,
        .offset = 0,
        .size = importanceBytes,
    };
    context.commandBuffer->barrier(BarrierDesc{
        .textures = &textureToTransfer,
        .textureCount = 1,
        .buffers = &bufferToTransfer,
        .bufferCount = 1,
    });

    context.commandBuffer->copyBufferToTexture(BufferTextureCopyDesc{
        .buffer = staging->radiance.get(),
        .texture = next->radiance.get(),
        .width = next->width,
        .height = next->height,
        .depth = 1,
    });
    context.commandBuffer->copyBuffer(BufferCopyDesc{
        .source = staging->importance.get(),
        .destination = next->importanceBuffer.get(),
        .size = importanceBytes,
    });
    context.host.retire(std::static_pointer_cast<void>(staging));

    TextureBarrierDesc textureToRead{
        .texture = next->radiance.get(),
        .before = ResourceState::TransferDestination,
        .after = ResourceState::ShaderRead,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    BufferBarrierDesc bufferToRead{
        .buffer = next->importanceBuffer.get(),
        .before = ResourceState::TransferDestination,
        .after = ResourceState::ShaderRead,
        .offset = 0,
        .size = importanceBytes,
    };
    context.commandBuffer->barrier(BarrierDesc{
        .textures = &textureToRead,
        .textureCount = 1,
        .buffers = &bufferToRead,
        .bufferCount = 1,
    });
    if (pdfCompute_.valid()) {
        result = pdfCompute_.buildEnvironment(
            *context.commandBuffer,
            *next->radianceView,
            next->pdf);
        if (!result) {
            log = "EnvironmentLightingSubsystem failed to build the environment PDF";
            return result;
        }
    }

    if (resources_ != nullptr) {
        context.host.retire(std::static_pointer_cast<void>(resources_));
    }
    resources_ = std::move(next);
    ++resourceRevision_;
    snapshot_.status = decoded.placeholder
        ? EnvironmentLightingStatus::Loading
        : (decoded.error.empty()
            ? EnvironmentLightingStatus::Ready
            : EnvironmentLightingStatus::Degraded);
    snapshot_.error = std::move(decoded.error);
    refreshSnapshot();
    return {};
}

void EnvironmentLightingSubsystem::refreshSnapshot()
{
    snapshot_.settings = requestedSettings_;
    snapshot_.settingsRevision = requestedSettingsRevision_;
    snapshot_.resourceRevision = resourceRevision_;
    if (resources_ == nullptr) {
        snapshot_.radianceView = nullptr;
        snapshot_.importanceBuffer = nullptr;
        snapshot_.pdfView = nullptr;
        snapshot_.sphericalHarmonics = nullptr;
        snapshot_.cpuImportanceTable = nullptr;
        snapshot_.width = 1;
        snapshot_.height = 1;
        snapshot_.importanceTexelCount = 1;
        snapshot_.mapAvailable = false;
        return;
    }
    snapshot_.radianceView = resources_->radianceView.get();
    snapshot_.importanceBuffer = resources_->importanceBuffer.get();
    snapshot_.pdfView = resources_->pdf.valid() ? resources_->pdf.view() : nullptr;
    snapshot_.sphericalHarmonics = &resources_->sphericalHarmonics;
    snapshot_.cpuImportanceTable = &resources_->cpuImportance;
    snapshot_.width = resources_->width;
    snapshot_.height = resources_->height;
    snapshot_.importanceTexelCount = static_cast<uint32_t>(resources_->cpuImportance.size());
    snapshot_.mapAvailable = resources_->mapAvailable;
}

void EnvironmentLightingSubsystem::shutdown()
{
    for (DecodeJob& job : decodeJobs_) {
        if (job.future.valid()) {
            job.future.wait();
        }
    }
    decodeJobs_.clear();
    pendingDecodePath_.clear();
    pendingDecodeGeneration_ = 0;
    readyDecode_.reset();
    resources_.reset();
    pdfCompute_.clear();
    snapshot_ = {};
    device_ = nullptr;
    world_ = nullptr;
    requestInitialized_ = false;
}

} // namespace metallic::render
