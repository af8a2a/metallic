#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/SlangCompiler.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
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

struct EnvironmentUploadResources {
    std::unique_ptr<Buffer> radiance;
    std::unique_ptr<Buffer> sphericalHarmonicsPartials;
};

constexpr uint32_t kEnvironmentSHCoefficientCount = 9;
constexpr uint32_t kEnvironmentSHThreadCount = 128;
constexpr uint32_t kEnvironmentSHMaxDispatchWidth = 65535;
constexpr uint64_t kEnvironmentSpecularBytes = 256ull * 128 * 8 * sizeof(std::array<float, 4>);

struct EnvironmentLightingPrecomputePush {
    uint32_t mode = 0;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t partialCount = 1;
    uint32_t dispatchWidth = 1;
    uint32_t procedural = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

static_assert(sizeof(EnvironmentLightingPrecomputePush) == 32);

} // namespace

struct EnvironmentLightingSubsystem::DecodedEnvironment {
    uint64_t generation = 0;
    std::vector<float> pixels;
    // Component offsets into pixels, including the unchanged full-resolution base.
    std::vector<size_t> mipOffsets{0};
    uint32_t width = 1;
    uint32_t height = 1;
    bool mapAvailable = false;
    bool placeholder = false;
    std::string error;

    void buildMipChain()
    {
        uint32_t sourceWidth = width, sourceHeight = height;
        while (sourceWidth > 1 || sourceHeight > 1) {
            const uint32_t targetWidth = std::max(sourceWidth / 2, 1u);
            const uint32_t targetHeight = std::max(sourceHeight / 2, 1u);
            const size_t sourceOffset = mipOffsets.back(), targetOffset = pixels.size();
            pixels.resize(targetOffset + size_t(targetWidth) * targetHeight * 4);
            // Integrate the covered spherical area, including fractional source
            // texels for NPOT images. An unweighted box overweights the poles.
            for (uint32_t y = 0; y < targetHeight; ++y) {
                const double y0 = double(y) * sourceHeight / targetHeight;
                const double y1 = double(y + 1) * sourceHeight / targetHeight;
                const uint32_t beginY = static_cast<uint32_t>(std::floor(y0));
                const uint32_t endY = std::min(static_cast<uint32_t>(std::ceil(y1)), sourceHeight);
                std::array<double, 3> latitudeWeights{};
                for (uint32_t sy = beginY; sy < endY; ++sy) {
                    latitudeWeights[sy - beginY] =
                        std::cos(std::numbers::pi * std::max(y0, double(sy)) / sourceHeight) -
                        std::cos(std::numbers::pi * std::min(y1, double(sy + 1)) / sourceHeight);
                }
                for (uint32_t x = 0; x < targetWidth; ++x) {
                    const double x0 = double(x) * sourceWidth / targetWidth;
                    const double x1 = double(x + 1) * sourceWidth / targetWidth;
                    const uint32_t beginX = static_cast<uint32_t>(std::floor(x0));
                    const uint32_t endX = std::min(static_cast<uint32_t>(std::ceil(x1)), sourceWidth);
                    std::array<double, 4> sum{};
                    double weightSum = 0.0;
                    for (uint32_t sy = beginY; sy < endY; ++sy) {
                        for (uint32_t sx = beginX; sx < endX; ++sx) {
                            const double weight = latitudeWeights[sy - beginY] *
                                (std::min(x1, double(sx + 1)) - std::max(x0, double(sx)));
                            const size_t index = sourceOffset + (size_t(sy) * sourceWidth + sx) * 4;
                            for (uint32_t channel = 0; channel < 4; ++channel) {
                                const float value = pixels[index + channel];
                                sum[channel] += (std::isfinite(value) ? std::max(value, 0.0f) : 0.0f) * weight;
                            }
                            weightSum += weight;
                        }
                    }
                    const size_t index = targetOffset + (size_t(y) * targetWidth + x) * 4;
                    for (uint32_t channel = 0; channel < 4; ++channel) {
                        pixels[index + channel] = static_cast<float>(sum[channel] / weightSum);
                    }
                }
            }
            mipOffsets.push_back(targetOffset);
            sourceWidth = targetWidth;
            sourceHeight = targetHeight;
        }
    }
};

struct EnvironmentLightingSubsystem::DecodeJob {
    std::future<DecodedEnvironment> future;
};

struct EnvironmentLightingSubsystem::GpuPrecompute {
    ComputeProgram program;

    Result initialize(Device& device, std::string& log)
    {
        ShaderCompileResult compileResult;
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = "Features/Environment/EnvironmentLightingPrecompute",
                .entryPointName = "environmentLightingPrecomputeMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            },
            compileResult);
        if (!result) {
            log = "EnvironmentLightingSubsystem failed to compile GPU SH precompute";
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            return result;
        }
        const std::array bindings{
            ComputeProgramBindingDesc{
                .binding = 0,
                .kind = ComputeResourceBindingKind::SampledImage,
            },
            ComputeProgramBindingDesc{
                .binding = 1,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeProgramBindingDesc{
                .binding = 2,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeProgramBindingDesc{
                .binding = 3,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
        };
        return program.initialize(
            device,
            ComputeProgramDesc{
                .spirv = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(EnvironmentLightingPrecomputePush),
                .bindings = bindings.data(),
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .debugName = "EnvironmentLightingPrecompute",
                .resourceTableCount = 3,
                .requiresRayQuery = false,
            },
            log);
    }

    Result build(
        CommandBuffer& commandBuffer,
        TextureView& radianceView,
        Buffer& partials,
        Buffer& coefficients,
        Buffer& specular,
        uint32_t width,
        uint32_t height,
        bool procedural)
    {
        const uint64_t texelCount = static_cast<uint64_t>(width) * height;
        const uint32_t partialCount = static_cast<uint32_t>(
            (texelCount + kEnvironmentSHThreadCount - 1u) / kEnvironmentSHThreadCount);
        const uint32_t dispatchWidth = std::min(partialCount, kEnvironmentSHMaxDispatchWidth);
        const uint32_t dispatchHeight =
            (partialCount + dispatchWidth - 1u) / dispatchWidth;
        TextureView* const radianceViews[] = {&radianceView};
        const std::array bindings{
            ComputeDispatchBinding{
                .binding = 0,
                .textureViews = radianceViews,
                .textureViewCount = static_cast<uint32_t>(std::size(radianceViews)),
            },
            ComputeDispatchBinding{.binding = 1, .buffer = &partials},
            ComputeDispatchBinding{.binding = 2, .buffer = &coefficients},
            ComputeDispatchBinding{.binding = 3, .buffer = &specular},
        };
        EnvironmentLightingPrecomputePush push{
            .width = width,
            .height = height,
            .partialCount = partialCount,
            .dispatchWidth = dispatchWidth,
            .procedural = procedural ? 1u : 0u,
        };
        Result result = program.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = bindings.data(),
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = dispatchWidth,
            .groupCountY = dispatchHeight,
            .groupCountZ = 1,
            .resourceTableIndex = 0,
        });
        if (!result) {
            return result;
        }
        BufferBarrierDesc partialsBarrier{
            .buffer = &partials,
            .before = ResourceState::General,
            .after = ResourceState::General,
            .offset = 0,
            .size = partials.desc().size,
        };
        commandBuffer.barrier(BarrierDesc{.buffers = &partialsBarrier, .bufferCount = 1});
        push.mode = 1;
        result = program.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = bindings.data(),
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = kEnvironmentSHCoefficientCount,
            .groupCountY = 1,
            .groupCountZ = 1,
            .resourceTableIndex = 1,
        });
        if (!result) { return result; }
        push.mode = 2;
        return program.dispatch({.commandBuffer = &commandBuffer,
            .bindings = bindings.data(), .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pushData = &push, .pushDataSize = sizeof(push),
            .groupCountX = 256 * 128 * 8 / kEnvironmentSHThreadCount,
            .resourceTableIndex = 2});
    }
};

class EnvironmentLightingSubsystem::ShaderReload final : public RenderSubsystemShaderReload {
public:
    ShaderReload(
        EnvironmentLightingSubsystem& owner,
        ImportancePdfCompute pdfCompute,
        std::unique_ptr<GpuPrecompute> gpuPrecompute)
        : owner_(owner)
        , pdfCompute_(std::move(pdfCompute))
        , gpuPrecompute_(std::move(gpuPrecompute))
    {
    }

    void commit() noexcept override
    {
        owner_.pdfCompute_ = std::move(pdfCompute_);
        owner_.gpuPrecompute_ = std::move(gpuPrecompute_);
        owner_.requestInitialized_ = false;
    }

private:
    EnvironmentLightingSubsystem& owner_;
    ImportancePdfCompute pdfCompute_;
    std::unique_ptr<GpuPrecompute> gpuPrecompute_;
};

struct EnvironmentLightingSubsystem::Resources {
    std::unique_ptr<Texture> radiance;
    std::unique_ptr<TextureView> radianceView;
    ImportancePdfTexture pdf;
    std::unique_ptr<Buffer> sphericalHarmonicsBuffer;
    std::unique_ptr<Buffer> prefilteredSpecularBuffer;
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
    Result result = pdfCompute_.initialize(context.device, log);
    if (!result) {
        return result;
    }
    gpuPrecompute_ = std::make_unique<GpuPrecompute>();
    return gpuPrecompute_->initialize(context.device, log);
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
        readyDecode_ = std::move(decoded);
        return;
    }

    if (resources_ == nullptr) {
        auto placeholder = std::make_unique<DecodedEnvironment>();
        placeholder->generation = generation;
        placeholder->pixels = {0.0f, 0.0f, 0.0f, 1.0f};
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
        decoded.buildMipChain();
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
    if (pendingPublication_ != nullptr && !pendingPublication_->resolved()) {
        log = "Environment publication must be submitted or cancelled before recording again";
        return makeError(Error::InvalidArgument);
    }
    if (readyDecode_ == nullptr) { return {}; }
    const auto decoded = readyDecode_;
    const auto previousResources = resources_;
    const auto previousSnapshot = snapshot_;
    const auto previousRevision = resourceRevision_;
    Result result = context.host.deferSubmission(*context.commandBuffer, {},
        [this, decoded, previousResources, previousSnapshot, previousRevision]() {
            resources_ = previousResources;
            resourceRevision_ = previousRevision;
            if (decoded->generation == requestedGeneration_) {
                snapshot_.status = previousSnapshot.status;
                snapshot_.error = previousSnapshot.error;
                // A completed async decode may have replaced this placeholder.
                if (readyDecode_ == nullptr) { readyDecode_ = decoded; }
            }
            refreshSnapshot();
        }, &pendingPublication_);
    if (!result) { return result; }
    result = publishDecoded(context, *decoded, log);
    if (result && readyDecode_ == decoded) { readyDecode_.reset(); }
    return result;
}

Result EnvironmentLightingSubsystem::prepareShaderReload(
    const RenderSubsystemInitContext& context,
    std::unique_ptr<RenderSubsystemShaderReload>& outReload,
    std::string& log)
{
    outReload.reset();
    if (device_ != &context.device) {
        log = "EnvironmentLightingSubsystem belongs to another Device";
        return makeError(Error::InvalidArgument);
    }

    ImportancePdfCompute nextPdfCompute;
    Result result = nextPdfCompute.initialize(context.device, log);
    if (!result) {
        return result;
    }
    auto nextGpuPrecompute = std::make_unique<GpuPrecompute>();
    result = nextGpuPrecompute->initialize(context.device, log);
    if (!result) {
        return result;
    }

    outReload = std::make_unique<ShaderReload>(
        *this,
        std::move(nextPdfCompute),
        std::move(nextGpuPrecompute));
    log = "reloaded environment importance and spherical-harmonics shaders";
    return {};
}

Result EnvironmentLightingSubsystem::publishDecoded(
    const RenderSubsystemFrameContext& context,
    const DecodedEnvironment& decoded,
    std::string& log)
{
    if (decoded.generation != requestedGeneration_ ||
        device_ == nullptr ||
        gpuPrecompute_ == nullptr ||
        !pdfCompute_.valid()) {
        return {};
    }
    if (decoded.pixels.empty()) {
        return makeError(Error::InvalidArgument);
    }

    auto next = std::make_shared<Resources>();
    next->width = std::max(decoded.width, 1u);
    next->height = std::max(decoded.height, 1u);
    next->mapAvailable = decoded.mapAvailable;
    const auto mipCount = static_cast<uint32_t>(decoded.mipOffsets.size());

    Result result = device_->createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
            .format = Format::Rgba32Sfloat,
            .width = next->width,
            .height = next->height,
            .depth = 1,
            .mipCount = mipCount,
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
            .mipCount = mipCount,
            .baseLayer = 0,
            .layerCount = 1,
        },
        next->radianceView);
    if (!result || next->radianceView == nullptr) {
        log = "EnvironmentLightingSubsystem createTextureView returned " + std::string(resultToString(result));
        return result ? makeError(Error::Failure) : result;
    }
    constexpr uint64_t kSphericalHarmonicsBytes =
        kEnvironmentSHCoefficientCount * sizeof(std::array<float, 4>);
    result = device_->createBuffer({.size = kEnvironmentSpecularBytes,
        .structureStride = sizeof(std::array<float, 4>), .usage = BufferUsageBits::Storage,
        .memoryLocation = MemoryLocation::Device,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}, next->prefilteredSpecularBuffer);
    if (!result) { log = "Environment specular prefilter allocation failed"; return result; }
    result = device_->createBuffer(
        BufferDesc{
            .size = kSphericalHarmonicsBytes,
            .structureStride = sizeof(std::array<float, 4>),
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
        },
        next->sphericalHarmonicsBuffer);
    if (!result || next->sphericalHarmonicsBuffer == nullptr) {
        log = "EnvironmentLightingSubsystem failed to create the GPU SH buffer";
        return result ? makeError(Error::Failure) : result;
    }
    result = next->pdf.initialize(
        *device_,
        next->width,
        next->height,
        "EnvironmentLightingSubsystem PDF",
        log);
    if (!result) {
        return result;
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
    const uint32_t shWidth = decoded.mapAvailable ? next->width : 256u;
    const uint32_t shHeight = decoded.mapAvailable ? next->height : 128u;
    const uint64_t texelCount = static_cast<uint64_t>(shWidth) * shHeight;
    const uint64_t partialCount =
        (texelCount + kEnvironmentSHThreadCount - 1u) / kEnvironmentSHThreadCount;
    const uint64_t partialBytes = partialCount * kSphericalHarmonicsBytes;
    result = device_->createBuffer(
        BufferDesc{
            .size = partialBytes,
            .structureStride = sizeof(std::array<float, 4>),
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        },
        staging->sphericalHarmonicsPartials);
    if (!result || staging->sphericalHarmonicsPartials == nullptr) {
        log = "EnvironmentLightingSubsystem failed to create the GPU SH partial buffer";
        return result ? makeError(Error::Failure) : result;
    }
    void* mappedRadiance = staging->radiance->map();
    if (mappedRadiance == nullptr) {
        log = "EnvironmentLightingSubsystem failed to map its radiance staging buffer";
        return makeError(Error::Failure);
    }
    std::memcpy(mappedRadiance, decoded.pixels.data(), static_cast<size_t>(radianceBytes));
    staging->radiance->flush(0, radianceBytes);
    staging->radiance->unmap();

    // Keep every referenced allocation alive even if a later recording step fails.
    context.host.retire(std::static_pointer_cast<void>(staging));
    context.host.retire(std::static_pointer_cast<void>(next));
    TextureBarrierDesc textureToTransfer{
        .texture = next->radiance.get(),
        .before = ResourceState::Undefined,
        .after = ResourceState::TransferDestination,
        .baseMip = 0,
        .mipCount = mipCount,
        .baseLayer = 0,
        .layerCount = 1,
    };
    std::array precomputeToGeneral{
        BufferBarrierDesc{
            .buffer = next->prefilteredSpecularBuffer.get(),
            .before = ResourceState::Undefined, .after = ResourceState::General,
        },
        BufferBarrierDesc{
            .buffer = staging->sphericalHarmonicsPartials.get(),
            .before = ResourceState::Undefined,
            .after = ResourceState::General,
            .offset = 0,
            .size = partialBytes,
        },
        BufferBarrierDesc{
            .buffer = next->sphericalHarmonicsBuffer.get(),
            .before = ResourceState::Undefined,
            .after = ResourceState::General,
            .offset = 0,
            .size = kSphericalHarmonicsBytes,
        },
    };
    context.commandBuffer->barrier(BarrierDesc{
        .buffers = precomputeToGeneral.data(),
        .bufferCount = static_cast<uint32_t>(precomputeToGeneral.size()),
    });
    context.commandBuffer->barrier(BarrierDesc{
        .textures = &textureToTransfer,
        .textureCount = 1,
    });

    for (uint32_t mip = 0; mip < mipCount; ++mip) {
        context.commandBuffer->copyBufferToTexture(BufferTextureCopyDesc{
            .buffer = staging->radiance.get(),
            .texture = next->radiance.get(),
            .bufferOffset = decoded.mipOffsets[mip] * sizeof(float),
            .width = std::max(next->width >> mip, 1u),
            .height = std::max(next->height >> mip, 1u),
            .depth = 1,
            .mipLevel = mip,
        });
    }

    TextureBarrierDesc textureToRead{
        .texture = next->radiance.get(),
        .before = ResourceState::TransferDestination,
        .after = ResourceState::ShaderRead,
        .baseMip = 0,
        .mipCount = mipCount,
        .baseLayer = 0,
        .layerCount = 1,
    };
    context.commandBuffer->barrier(BarrierDesc{
        .textures = &textureToRead,
        .textureCount = 1,
    });
    result = pdfCompute_.buildEnvironment(
        *context.commandBuffer,
        *next->radianceView,
        next->pdf);
    if (!result) {
        log = "EnvironmentLightingSubsystem failed to build the environment PDF on the GPU";
        return result;
    }
    result = gpuPrecompute_->build(
        *context.commandBuffer,
        *next->radianceView,
        *staging->sphericalHarmonicsPartials,
        *next->sphericalHarmonicsBuffer,
        *next->prefilteredSpecularBuffer,
        shWidth,
        shHeight,
        !decoded.mapAvailable);
    if (!result) {
        log = "EnvironmentLightingSubsystem failed to integrate environment SH on the GPU: ";
        log += resultToString(result);
        return result;
    }
    BufferBarrierDesc sphericalHarmonicsToRead{
        .buffer = next->sphericalHarmonicsBuffer.get(),
        .before = ResourceState::General,
        .after = ResourceState::ShaderRead,
        .offset = 0,
        .size = kSphericalHarmonicsBytes,
    };
    context.commandBuffer->barrier(BarrierDesc{
        .buffers = &sphericalHarmonicsToRead,
        .bufferCount = 1,
    });
    BufferBarrierDesc specularToRead{.buffer = next->prefilteredSpecularBuffer.get(),
        .before = ResourceState::General, .after = ResourceState::ShaderRead};
    context.commandBuffer->barrier({.buffers = &specularToRead, .bufferCount = 1});
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
    snapshot_.error = decoded.error;
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
        snapshot_.pdfView = nullptr;
        snapshot_.sphericalHarmonicsBuffer = nullptr;
        snapshot_.prefilteredSpecularBuffer = nullptr;
        snapshot_.width = 1;
        snapshot_.height = 1;
        snapshot_.mapAvailable = false;
        return;
    }
    snapshot_.radianceView = resources_->radianceView.get();
    snapshot_.pdfView = resources_->pdf.valid() ? resources_->pdf.view() : nullptr;
    snapshot_.sphericalHarmonicsBuffer = resources_->sphericalHarmonicsBuffer.get();
    snapshot_.prefilteredSpecularBuffer = resources_->prefilteredSpecularBuffer.get();
    snapshot_.width = resources_->width;
    snapshot_.height = resources_->height;
    snapshot_.mapAvailable = resources_->mapAvailable;
}

void EnvironmentLightingSubsystem::shutdown()
{
    if (pendingPublication_ != nullptr) { pendingPublication_->cancel(); }
    pendingPublication_.reset();
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
    gpuPrecompute_.reset();
    snapshot_ = {};
    device_ = nullptr;
    world_ = nullptr;
    requestInitialized_ = false;
}

} // namespace metallic::render
