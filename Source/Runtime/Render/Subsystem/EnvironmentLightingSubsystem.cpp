#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/SlangCompiler.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <chrono>
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

struct EnvironmentUploadResources {
    std::unique_ptr<Buffer> radiance;
    std::unique_ptr<Buffer> sphericalHarmonicsPartials;
};

constexpr uint32_t kEnvironmentSHCoefficientCount = 9;
constexpr uint32_t kEnvironmentSHThreadCount = 128;
constexpr uint32_t kEnvironmentSHMaxDispatchWidth = 65535;

struct EnvironmentLightingPrecomputePush {
    uint32_t mode = 0;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t partialCount = 1;
    uint32_t dispatchWidth = 1;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

static_assert(sizeof(EnvironmentLightingPrecomputePush) == 32);

} // namespace

struct EnvironmentLightingSubsystem::DecodedEnvironment {
    uint64_t generation = 0;
    std::vector<float> pixels;
    uint32_t width = 1;
    uint32_t height = 1;
    bool mapAvailable = false;
    bool placeholder = false;
    std::string error;
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
                .moduleName = "EnvironmentLightingPrecompute",
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
                .descriptorSetCount = 2,
                .requiresRayQuery = false,
            },
            log);
    }

    Result build(
        CommandBuffer& commandBuffer,
        TextureView& radianceView,
        Buffer& partials,
        Buffer& coefficients,
        uint32_t width,
        uint32_t height)
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
        };
        EnvironmentLightingPrecomputePush push{
            .width = width,
            .height = height,
            .partialCount = partialCount,
            .dispatchWidth = dispatchWidth,
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
            .descriptorSetIndex = 0,
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
        return program.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = bindings.data(),
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = kEnvironmentSHCoefficientCount,
            .groupCountY = 1,
            .groupCountZ = 1,
            .descriptorSetIndex = 1,
        });
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
    if (readyDecode_ == nullptr) {
        return {};
    }
    return publishDecoded(context, std::move(*readyDecode_), log);
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
    DecodedEnvironment decoded,
    std::string& log)
{
    readyDecode_.reset();
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
    constexpr uint64_t kSphericalHarmonicsBytes =
        kEnvironmentSHCoefficientCount * sizeof(std::array<float, 4>);
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
    const uint64_t texelCount = static_cast<uint64_t>(next->width) * next->height;
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

    TextureBarrierDesc textureToTransfer{
        .texture = next->radiance.get(),
        .before = ResourceState::Undefined,
        .after = ResourceState::TransferDestination,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    std::array precomputeToGeneral{
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

    context.commandBuffer->copyBufferToTexture(BufferTextureCopyDesc{
        .buffer = staging->radiance.get(),
        .texture = next->radiance.get(),
        .width = next->width,
        .height = next->height,
        .depth = 1,
    });

    TextureBarrierDesc textureToRead{
        .texture = next->radiance.get(),
        .before = ResourceState::TransferDestination,
        .after = ResourceState::ShaderRead,
        .baseMip = 0,
        .mipCount = 1,
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
        next->width,
        next->height);
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
    context.host.retire(std::static_pointer_cast<void>(staging));

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
        snapshot_.pdfView = nullptr;
        snapshot_.sphericalHarmonicsBuffer = nullptr;
        snapshot_.width = 1;
        snapshot_.height = 1;
        snapshot_.mapAvailable = false;
        return;
    }
    snapshot_.radianceView = resources_->radianceView.get();
    snapshot_.pdfView = resources_->pdf.valid() ? resources_->pdf.view() : nullptr;
    snapshot_.sphericalHarmonicsBuffer = resources_->sphericalHarmonicsBuffer.get();
    snapshot_.width = resources_->width;
    snapshot_.height = resources_->height;
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
    gpuPrecompute_.reset();
    snapshot_ = {};
    device_ = nullptr;
    world_ = nullptr;
    requestInitialized_ = false;
}

} // namespace metallic::render
