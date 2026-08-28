#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetConfig.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "openpbr_data_constants.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace metallic::render::builtin_pass {
namespace {

using GPUDrivenCompileClock = std::chrono::steady_clock;

void logGPUDrivenCompileStage(
    std::string_view stage,
    GPUDrivenCompileClock::time_point begin)
{
    const double elapsedMilliseconds = std::chrono::duration<double, std::milli>(
        GPUDrivenCompileClock::now() - begin).count();
    spdlog::info(
        "[GPUDrivenPreviewPass] Compile stage '{}' completed in {:.2f} ms",
        stage,
        elapsedMilliseconds);
}

const char* pipelineCacheLoadStatusName(PipelineCacheLoadStatus status)
{
    switch (status) {
    case PipelineCacheLoadStatus::NotFound: return "not-found";
    case PipelineCacheLoadStatus::Loaded: return "loaded";
    case PipelineCacheLoadStatus::Invalid: return "invalid";
    case PipelineCacheLoadStatus::Incompatible: return "incompatible";
    }
    return "unknown";
}

constexpr uint32_t kGPUDrivenMaxMaterialTextures = 256;
constexpr uint32_t kGPUDrivenEnvironmentModeProcedural = 0;
constexpr uint32_t kGPUDrivenEnvironmentModeMap = 1;
constexpr uint32_t kGPUDrivenEnvironmentModeDisabled = 2;
constexpr uint32_t kGPUDrivenOpenPBRLut2DCount = 6;
constexpr uint32_t kGPUDrivenOpenPBRLut3DCount = 2;
constexpr uint32_t kGPUDrivenOpenPBRLutSize = OpenPBR_EnergyTableSize;
constexpr uint32_t kGPUDrivenOpenPBRLtcSize = OpenPBR_LTCTableSize;
constexpr float kGPUDrivenOpenPBRLutScale = 1.0f / 65535.0f;
constexpr uint32_t kGPUDrivenInvalidBindlessIndex =
    std::numeric_limits<uint32_t>::max();
constexpr const char* kGPUDrivenPipelineCachePath =
    PROJECT_SOURCE_DIR "/.cache/pso/GPUDrivenPreviewPass.pso";

bool previewStreamEnabled(const RenderGraphProperties& properties)
{
    const auto enabled = properties.find("enableMeshletStreaming");
    if (enabled != properties.end() && enabled->is_boolean()) {
        return enabled->get<bool>();
    }
    const auto assetPath = properties.find("streamAssetPath");
    return assetPath != properties.end() && assetPath->is_string();
}

uint32_t previewStreamUintProperty(
    const RenderGraphProperties& properties,
    const char* key,
    uint32_t fallback)
{
    const auto iter = properties.find(key);
    if (iter == properties.end() || !iter->is_number_integer()) {
        return fallback;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 || value > std::numeric_limits<uint32_t>::max()
        ? fallback
        : static_cast<uint32_t>(value);
}

uint64_t previewStreamUint64Property(
    const RenderGraphProperties& properties,
    const char* key,
    uint64_t fallback)
{
    const auto iter = properties.find(key);
    if (iter == properties.end() || !iter->is_number_integer()) {
        return fallback;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 ? fallback : static_cast<uint64_t>(value);
}

std::filesystem::path previewStreamAssetPath(
    const RenderGraphProperties& properties,
    const std::filesystem::path& sourcePath)
{
    const auto iter = properties.find("streamAssetPath");
    if (iter == properties.end() || !iter->is_string()) {
        return scene::meshletStreamAssetPathFor(sourcePath);
    }
    std::filesystem::path path = iter->get<std::string>();
    return path.is_relative()
        ? std::filesystem::path(PROJECT_SOURCE_DIR) / path
        : path;
}

struct PreviewStreamSourceSelection {
    std::string sourceId;
    std::filesystem::path sourcePath;
};

bool resolvePreviewStreamSource(
    const RenderGraphProperties& properties,
    const scene::Scene* runtimeScene,
    const std::filesystem::path& scenePath,
    PreviewStreamSourceSelection& outSelection,
    std::string& log)
{
    outSelection = {};
    if (runtimeScene == nullptr) {
        log = "GPUDrivenPreviewPass stream integration requires a runtime scene";
        return false;
    }

    std::string requestedSourceId;
    const auto sourceIdProperty = properties.find("streamSourceId");
    if (sourceIdProperty != properties.end()) {
        if (!sourceIdProperty->is_string() ||
            sourceIdProperty->get_ref<const std::string&>().empty()) {
            log = "GPUDrivenPreviewPass streamSourceId must be a non-empty string";
            return false;
        }
        requestedSourceId = sourceIdProperty->get<std::string>();
    }

    const std::vector<scene::SceneSourceDesc>& sources = runtimeScene->sources();
    if (sources.empty()) {
        if (!requestedSourceId.empty()) {
            log = "GPUDrivenPreviewPass streamSourceId was provided for a non-composed scene";
            return false;
        }
        outSelection.sourcePath = scenePath;
        return true;
    }

    const auto selectSource = [&](const scene::SceneSourceDesc& source) {
        outSelection.sourceId = source.id;
        outSelection.sourcePath = source.path;
    };
    if (!requestedSourceId.empty()) {
        const auto source = std::find_if(
            sources.begin(),
            sources.end(),
            [&](const scene::SceneSourceDesc& candidate) {
                return candidate.id == requestedSourceId;
            });
        if (source == sources.end()) {
            log = "GPUDrivenPreviewPass streamSourceId '" + requestedSourceId +
                "' does not name a composed scene source";
            return false;
        }
        selectSource(*source);
        return true;
    }

    if (sources.size() == 1) {
        selectSource(sources.front());
        return true;
    }

    std::vector<const scene::SceneSourceDesc*> matches;
    const std::filesystem::path normalizedRequestedPath = normalizedScenePath(scenePath);
    for (const scene::SceneSourceDesc& source : sources) {
        if (normalizedScenePath(source.path) == normalizedRequestedPath) {
            matches.push_back(&source);
        }
    }
    if (matches.size() == 1) {
        selectSource(*matches.front());
        return true;
    }

    matches.clear();
    const auto assetPathProperty = properties.find("streamAssetPath");
    if (assetPathProperty != properties.end() && assetPathProperty->is_string()) {
        scene::MeshletStreamAsset candidateAsset;
        std::string reason;
        const std::filesystem::path assetPath =
            previewStreamAssetPath(properties, scenePath);
        if (!candidateAsset.open(assetPath, reason)) {
            log = "GPUDrivenPreviewPass cannot inspect streamAssetPath '" +
                assetPath.string() + "': " + reason;
            return false;
        }
        for (const scene::SceneSourceDesc& source : sources) {
            if (candidateAsset.isCurrentForSource(source.path)) {
                matches.push_back(&source);
            }
        }
    }
    if (matches.size() == 1) {
        selectSource(*matches.front());
        return true;
    }

    log = matches.empty()
        ? "GPUDrivenPreviewPass could not uniquely match the stream asset to a composed scene source; set streamSourceId"
        : "GPUDrivenPreviewPass stream asset matches multiple composed scene sources; set streamSourceId to disambiguate the owner";
    return false;
}

MeshletStreamRuntimeDesc previewStreamRuntimeDesc(
    const RenderGraphProperties& properties,
    const std::filesystem::path& sourcePath)
{
    const uint32_t maxGpuPageRequests = std::max(
        previewStreamUintProperty(
            properties,
            "maxGpuPageRequests",
            kMeshletStreamDefaultMaxGpuPageRequests),
        1u);
    return MeshletStreamRuntimeDesc{
        .sourcePath = sourcePath,
        .streamAssetPath = previewStreamAssetPath(properties, sourcePath),
        .autoBuildStreamAsset = false,
        .maxResidentBytes = previewStreamUint64Property(
            properties,
            "maxResidentBytes",
            0),
        .maxResidentPages = previewStreamUintProperty(
            properties,
            "maxResidentPages",
            4096),
        .maxLockedFallbackPages = previewStreamUintProperty(
            properties,
            "maxLockedFallbackPages",
            1024),
        .maxPageUploadsPerFrame = previewStreamUintProperty(
            properties,
            "maxPageUploadsPerFrame",
            64),
        .maxGpuPageRequests = maxGpuPageRequests,
        .maxGpuPageUnloadRequests = std::max(
            previewStreamUintProperty(
                properties,
                "maxGpuPageUnloadRequests",
                maxGpuPageRequests),
            1u),
        .maxActiveGroups = std::max(
            previewStreamUintProperty(
                properties,
                "maxActiveGroups",
                kMeshletStreamDefaultMaxActiveGroups),
            1u),
        .maxTraversalWorkers = std::max(
            previewStreamUintProperty(
                properties,
                "maxTraversalWorkers",
                kMeshletStreamDefaultTraversalWorkers),
            1u),
        .maxTraversalWorkItems = std::min(
            std::max(
                previewStreamUintProperty(
                    properties,
                    "maxTraversalWorkItems",
                    kMeshletStreamDefaultTraversalWorkItems),
                1u),
            kMeshletStreamMaxTraversalWorkItems),
        .pageLoadConcurrency = pageLoadConcurrencyFromProperties(properties),
        .maxPageLoadsInFlight = std::max(
            previewStreamUintProperty(
                properties,
                "maxPageLoadsInFlight",
                128),
            1u),
        .queuedFrameCount = 3,
        .enableClusterRtx = false,
    };
}

using GPUDrivenOpenPBRLutScalar = uint16_t;

struct GPUDrivenOpenPBRVec3 {
    float x;
    float y;
    float z;
};

constexpr GPUDrivenOpenPBRVec3 vec3(float x, float y, float z)
{
    return GPUDrivenOpenPBRVec3{x, y, z};
}

static constexpr GPUDrivenOpenPBRLutScalar kGPUDrivenOpenPBRIdealDielectricEnergyComplement[] = {
#include "impl/data/openpbr_ideal_dielectric_energy_complement_data.h"
};

static constexpr GPUDrivenOpenPBRLutScalar kGPUDrivenOpenPBRIdealDielectricAverageEnergyComplement[] = {
#include "impl/data/openpbr_ideal_dielectric_avg_energy_complement_data.h"
};

static constexpr GPUDrivenOpenPBRLutScalar kGPUDrivenOpenPBRIdealDielectricReflectionRatio[] = {
#include "impl/data/openpbr_ideal_dielectric_reflection_ratio_data.h"
};

static constexpr GPUDrivenOpenPBRLutScalar kGPUDrivenOpenPBROpaqueDielectricEnergyComplement[] = {
#include "impl/data/openpbr_opaque_dielectric_energy_complement_data.h"
};

static constexpr GPUDrivenOpenPBRLutScalar kGPUDrivenOpenPBROpaqueDielectricAverageEnergyComplement[] = {
#include "impl/data/openpbr_opaque_dielectric_avg_energy_complement_data.h"
};

static constexpr GPUDrivenOpenPBRLutScalar kGPUDrivenOpenPBRIdealMetalEnergyComplement[] = {
#include "impl/data/openpbr_ideal_metal_energy_complement_data.h"
};

static constexpr GPUDrivenOpenPBRLutScalar kGPUDrivenOpenPBRIdealMetalAverageEnergyComplement[] = {
#include "impl/data/openpbr_ideal_metal_avg_energy_complement_data.h"
};

static constexpr GPUDrivenOpenPBRVec3 kGPUDrivenOpenPBRLtc[] = {
#include "impl/data/openpbr_ltc_data.h"
};

static_assert(std::size(kGPUDrivenOpenPBRIdealDielectricEnergyComplement) ==
    kGPUDrivenOpenPBRLutSize * kGPUDrivenOpenPBRLutSize * kGPUDrivenOpenPBRLutSize);
static_assert(std::size(kGPUDrivenOpenPBRIdealDielectricAverageEnergyComplement) ==
    kGPUDrivenOpenPBRLutSize * kGPUDrivenOpenPBRLutSize);
static_assert(std::size(kGPUDrivenOpenPBRIdealDielectricReflectionRatio) ==
    kGPUDrivenOpenPBRLutSize * kGPUDrivenOpenPBRLutSize);
static_assert(std::size(kGPUDrivenOpenPBROpaqueDielectricEnergyComplement) ==
    kGPUDrivenOpenPBRLutSize * kGPUDrivenOpenPBRLutSize * kGPUDrivenOpenPBRLutSize);
static_assert(std::size(kGPUDrivenOpenPBROpaqueDielectricAverageEnergyComplement) ==
    kGPUDrivenOpenPBRLutSize * kGPUDrivenOpenPBRLutSize);
static_assert(std::size(kGPUDrivenOpenPBRIdealMetalEnergyComplement) ==
    kGPUDrivenOpenPBRLutSize * kGPUDrivenOpenPBRLutSize);
static_assert(std::size(kGPUDrivenOpenPBRIdealMetalAverageEnergyComplement) == kGPUDrivenOpenPBRLutSize);
static_assert(std::size(kGPUDrivenOpenPBRLtc) == kGPUDrivenOpenPBRLtcSize * kGPUDrivenOpenPBRLtcSize);

struct GPUDrivenPreviewTextureResource {
    std::unique_ptr<Buffer> uploadBuffer;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
    Format format = Format::Rgba8Unorm;
    ResourceState state = ResourceState::Undefined;
    bool uploaded = false;
};

struct GPUDrivenPreviewDecodedImage {
    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
};

std::string gpuDrivenResultMessage(std::string_view label, const Result& result)
{
    return std::string(label) + " returned " + resultToString(result);
}

Result createGPUDrivenTexture(
    Device& device,
    const void* pixels,
    uint64_t byteSize,
    uint32_t width,
    uint32_t height,
    Format format,
    std::string_view label,
    GPUDrivenPreviewTextureResource& outTexture,
    std::string& log,
    uint32_t depth = 1)
{
    if (pixels == nullptr || byteSize == 0 || width == 0 || height == 0 || depth == 0) {
        return makeError(Error::InvalidArgument);
    }

    GPUDrivenPreviewTextureResource texture;
    texture.width = width;
    texture.height = height;
    texture.depth = depth;
    texture.format = format;
    Result result = device.createBuffer(
        BufferDesc{
            .size = byteSize,
            .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy,
        },
        texture.uploadBuffer);
    if (!result || texture.uploadBuffer == nullptr) {
        log += gpuDrivenResultMessage(std::string("createBuffer(") + std::string(label) + " upload)", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    void* mapped = texture.uploadBuffer->map();
    if (mapped == nullptr) {
        log += std::string(label) + " upload buffer map failed\n";
        return makeError(Error::Failure);
    }
    std::memcpy(mapped, pixels, static_cast<size_t>(byteSize));
    texture.uploadBuffer->flush(0, byteSize);
    texture.uploadBuffer->unmap();

    result = device.createTexture(
        TextureDesc{
            .type = depth > 1 ? TextureType::Texture3D : TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
            .format = format,
            .width = width,
            .height = height,
            .depth = depth,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy,
        },
        texture.texture);
    if (!result || texture.texture == nullptr) {
        log += gpuDrivenResultMessage(std::string("createTexture(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    result = device.createTextureView(
        *texture.texture,
        TextureViewDesc{
            .format = format,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        },
        texture.view);
    if (!result || texture.view == nullptr) {
        log += gpuDrivenResultMessage(std::string("createTextureView(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    outTexture = std::move(texture);
    return {};
}

template <size_t ValueCount>
Result createGPUDrivenOpenPBRScalarLut(
    Device& device,
    const GPUDrivenOpenPBRLutScalar (&values)[ValueCount],
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    std::string_view label,
    GPUDrivenPreviewTextureResource& outTexture,
    std::string& log)
{
    const uint64_t texelCount =
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * static_cast<uint64_t>(depth);
    if (texelCount != ValueCount) {
        return makeError(Error::InvalidArgument);
    }
    std::vector<float> pixels(static_cast<size_t>(texelCount) * 4u, 0.0f);
    for (size_t index = 0; index < static_cast<size_t>(texelCount); ++index) {
        pixels[index * 4u] = static_cast<float>(values[index]) * kGPUDrivenOpenPBRLutScale;
        pixels[index * 4u + 3u] = 1.0f;
    }
    return createGPUDrivenTexture(
        device,
        pixels.data(),
        static_cast<uint64_t>(pixels.size() * sizeof(float)),
        width,
        height,
        Format::Rgba32Sfloat,
        label,
        outTexture,
        log,
        depth);
}

Result prepareGPUDrivenOpenPBRLuts(
    Device& device,
    std::array<GPUDrivenPreviewTextureResource, kGPUDrivenOpenPBRLut2DCount>& lut2D,
    std::array<GPUDrivenPreviewTextureResource, kGPUDrivenOpenPBRLut3DCount>& lut3D,
    std::string& log)
{
    Result result = createGPUDrivenOpenPBRScalarLut(
        device,
        kGPUDrivenOpenPBRIdealDielectricAverageEnergyComplement,
        kGPUDrivenOpenPBRLutSize,
        kGPUDrivenOpenPBRLutSize,
        1,
        "GPUDriven OpenPBR ideal dielectric average energy complement LUT",
        lut2D[0],
        log);
    if (!result) {
        return result;
    }
    result = createGPUDrivenOpenPBRScalarLut(
        device,
        kGPUDrivenOpenPBRIdealDielectricReflectionRatio,
        kGPUDrivenOpenPBRLutSize,
        kGPUDrivenOpenPBRLutSize,
        1,
        "GPUDriven OpenPBR ideal dielectric reflection ratio LUT",
        lut2D[1],
        log);
    if (!result) {
        return result;
    }
    result = createGPUDrivenOpenPBRScalarLut(
        device,
        kGPUDrivenOpenPBROpaqueDielectricAverageEnergyComplement,
        kGPUDrivenOpenPBRLutSize,
        kGPUDrivenOpenPBRLutSize,
        1,
        "GPUDriven OpenPBR opaque dielectric average energy complement LUT",
        lut2D[2],
        log);
    if (!result) {
        return result;
    }
    result = createGPUDrivenOpenPBRScalarLut(
        device,
        kGPUDrivenOpenPBRIdealMetalEnergyComplement,
        kGPUDrivenOpenPBRLutSize,
        kGPUDrivenOpenPBRLutSize,
        1,
        "GPUDriven OpenPBR ideal metal energy complement LUT",
        lut2D[3],
        log);
    if (!result) {
        return result;
    }
    result = createGPUDrivenOpenPBRScalarLut(
        device,
        kGPUDrivenOpenPBRIdealMetalAverageEnergyComplement,
        kGPUDrivenOpenPBRLutSize,
        1,
        1,
        "GPUDriven OpenPBR ideal metal average energy complement LUT",
        lut2D[4],
        log);
    if (!result) {
        return result;
    }

    std::vector<float> ltcPixels(std::size(kGPUDrivenOpenPBRLtc) * 4u, 0.0f);
    for (size_t index = 0; index < std::size(kGPUDrivenOpenPBRLtc); ++index) {
        ltcPixels[index * 4u] = kGPUDrivenOpenPBRLtc[index].x;
        ltcPixels[index * 4u + 1u] = kGPUDrivenOpenPBRLtc[index].y;
        ltcPixels[index * 4u + 2u] = kGPUDrivenOpenPBRLtc[index].z;
        ltcPixels[index * 4u + 3u] = 1.0f;
    }
    result = createGPUDrivenTexture(
        device,
        ltcPixels.data(),
        static_cast<uint64_t>(ltcPixels.size() * sizeof(float)),
        kGPUDrivenOpenPBRLtcSize,
        kGPUDrivenOpenPBRLtcSize,
        Format::Rgba32Sfloat,
        "GPUDriven OpenPBR LTC LUT",
        lut2D[5],
        log);
    if (!result) {
        return result;
    }
    result = createGPUDrivenOpenPBRScalarLut(
        device,
        kGPUDrivenOpenPBRIdealDielectricEnergyComplement,
        kGPUDrivenOpenPBRLutSize,
        kGPUDrivenOpenPBRLutSize,
        kGPUDrivenOpenPBRLutSize,
        "GPUDriven OpenPBR ideal dielectric energy complement LUT",
        lut3D[0],
        log);
    if (!result) {
        return result;
    }
    return createGPUDrivenOpenPBRScalarLut(
        device,
        kGPUDrivenOpenPBROpaqueDielectricEnergyComplement,
        kGPUDrivenOpenPBRLutSize,
        kGPUDrivenOpenPBRLutSize,
        kGPUDrivenOpenPBRLutSize,
        "GPUDriven OpenPBR opaque dielectric energy complement LUT",
        lut3D[1],
        log);
}

Result uploadGPUDrivenTexture(CommandBuffer& commandBuffer, GPUDrivenPreviewTextureResource& texture)
{
    if (texture.uploaded) {
        return {};
    }
    if (texture.uploadBuffer == nullptr || texture.texture == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    TextureBarrierDesc toTransfer{
        .texture = texture.texture.get(),
        .before = texture.state,
        .after = ResourceState::TransferDestination,
        .baseMip = 0,
        .mipCount = 1,
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer.barrier(BarrierDesc{
        .textures = &toTransfer,
        .textureCount = 1,
    });
    texture.state = ResourceState::TransferDestination;
    commandBuffer.copyBufferToTexture(BufferTextureCopyDesc{
        .buffer = texture.uploadBuffer.get(),
        .texture = texture.texture.get(),
        .width = texture.width,
        .height = texture.height,
        .depth = texture.depth,
        .mipLevel = 0,
        .baseLayer = 0,
    });
    TextureBarrierDesc toShaderRead{
        .texture = texture.texture.get(),
        .before = texture.state,
        .after = ResourceState::ShaderRead,
        .baseMip = 0,
        .mipCount = 1,
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

bool decodeGPUDrivenMaterialTexture(
    const scene::Scene& loadedScene,
    uint32_t textureIndex,
    GPUDrivenPreviewDecodedImage& outImage,
    std::string& log)
{
    outImage = GPUDrivenPreviewDecodedImage{};
    if (textureIndex >= loadedScene.textures().size()) {
        return false;
    }
    const scene::RenderTexture& texture = loadedScene.textures()[textureIndex];
    if (texture.imageIndex < 0 || static_cast<size_t>(texture.imageIndex) >= loadedScene.images().size()) {
        return false;
    }
    const scene::RenderImage& image = loadedScene.images()[static_cast<size_t>(texture.imageIndex)];
    if (!image.decodedMips.empty()) {
        const scene::RenderImage::Mip& mip = image.decodedMips.front();
        const uint64_t byteSize = static_cast<uint64_t>(mip.width) * mip.height * 4ull;
        if (mip.width == 0 || mip.height == 0 || mip.pixels.size() < byteSize) {
            return false;
        }
        outImage.width = mip.width;
        outImage.height = mip.height;
        outImage.pixels.assign(mip.pixels.begin(), mip.pixels.begin() + static_cast<size_t>(byteSize));
        return true;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels = nullptr;
    if (!image.encodedData.empty() && image.encodedData.size() <= static_cast<size_t>(std::numeric_limits<int>::max())) {
        pixels = stbi_load_from_memory(
            image.encodedData.data(),
            static_cast<int>(image.encodedData.size()),
            &width,
            &height,
            &channels,
            4);
    } else if (!image.uri.empty() && image.uri.rfind("data:", 0) != 0) {
        std::filesystem::path imagePath = image.uri;
        if (imagePath.is_relative()) {
            imagePath = loadedScene.filename().parent_path() / imagePath;
        }
        pixels = stbi_load(imagePath.string().c_str(), &width, &height, &channels, 4);
    }
    if (pixels == nullptr || width <= 0 || height <= 0) {
        log += "Warning: GPUDrivenPreviewPass failed to decode material texture ";
        log += texture.name.empty() ? image.name : texture.name;
        if (const char* reason = stbi_failure_reason()) {
            log += ": ";
            log += reason;
        }
        log += '\n';
        if (pixels != nullptr) {
            stbi_image_free(pixels);
        }
        return false;
    }
    const uint64_t byteSize = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ull;
    if (byteSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        stbi_image_free(pixels);
        return false;
    }
    outImage.width = static_cast<uint32_t>(width);
    outImage.height = static_cast<uint32_t>(height);
    outImage.pixels.assign(pixels, pixels + static_cast<size_t>(byteSize));
    stbi_image_free(pixels);
    return true;
}

GPUDrivenPreviewGpuTextureInfo gpuDrivenTextureInfo(
    const scene::RenderTextureInfo& textureInfo,
    const std::vector<uint32_t>& textureIndexMap)
{
    GPUDrivenPreviewGpuTextureInfo result;
    if (textureInfo.textureIndex >= 0 &&
        static_cast<size_t>(textureInfo.textureIndex) < textureIndexMap.size()) {
        result.textureIndex = textureIndexMap[static_cast<size_t>(textureInfo.textureIndex)];
    }
    result.texCoord = 0;
    result.transform0[0] = textureInfo.uvTransform[0];
    result.transform0[1] = textureInfo.uvTransform[1];
    result.transform0[2] = textureInfo.uvTransform[2];
    result.transform1[0] = textureInfo.uvTransform[3];
    result.transform1[1] = textureInfo.uvTransform[4];
    result.transform1[2] = textureInfo.uvTransform[5];
    return result;
}

GPUDrivenPreviewGpuMaterial gpuDrivenMaterial(
    const scene::RenderMaterial& material,
    const std::vector<uint32_t>& textureIndexMap)
{
    GPUDrivenPreviewGpuMaterial result;
    result.baseColor[0] = material.baseColorFactor.x;
    result.baseColor[1] = material.baseColorFactor.y;
    result.baseColor[2] = material.baseColorFactor.z;
    result.baseColor[3] = material.baseColorFactor.w;
    result.emissive[0] = material.emissiveFactor.x;
    result.emissive[1] = material.emissiveFactor.y;
    result.emissive[2] = material.emissiveFactor.z;
    result.params[0] = material.metallicFactor;
    result.params[1] = material.roughnessFactor;
    result.params[2] = material.alphaCutoff;
    result.params[3] = material.doubleSided ? 1.0f : 0.0f;
    result.textureParams[0] = material.normalTextureScale;
    result.textureParams[1] = material.occlusionTextureStrength;
    result.glassParams[0] = material.transmissionFactor;
    result.glassParams[1] = material.ior;
    result.glassParams[2] = material.thicknessFactor;
    result.glassParams[3] = material.attenuationDistance;
    result.attenuationColor[0] = material.attenuationColor.x;
    result.attenuationColor[1] = material.attenuationColor.y;
    result.attenuationColor[2] = material.attenuationColor.z;
    result.diffuseTransmission[0] = material.diffuseTransmissionColor.x;
    result.diffuseTransmission[1] = material.diffuseTransmissionColor.y;
    result.diffuseTransmission[2] = material.diffuseTransmissionColor.z;
    result.diffuseTransmission[3] = material.diffuseTransmissionFactor;
    result.baseColorTexture = gpuDrivenTextureInfo(material.baseColorTexture, textureIndexMap);
    result.metallicRoughnessTexture = gpuDrivenTextureInfo(material.metallicRoughnessTexture, textureIndexMap);
    result.normalTexture = gpuDrivenTextureInfo(material.normalTexture, textureIndexMap);
    result.occlusionTexture = gpuDrivenTextureInfo(material.occlusionTexture, textureIndexMap);
    result.emissiveTexture = gpuDrivenTextureInfo(material.emissiveTexture, textureIndexMap);
    result.transmissionTexture = gpuDrivenTextureInfo(material.transmissionTexture, textureIndexMap);
    result.thicknessTexture = gpuDrivenTextureInfo(material.thicknessTexture, textureIndexMap);
    result.diffuseTransmissionTexture = gpuDrivenTextureInfo(material.diffuseTransmissionTexture, textureIndexMap);
    result.diffuseTransmissionColorTexture = gpuDrivenTextureInfo(
        material.diffuseTransmissionColorTexture,
        textureIndexMap);
    return result;
}

struct GPUDrivenPreviewMeshletRange {
    uint32_t offset = 0;
    uint32_t count = 0;
};

struct GPUDrivenPreviewCullingTargets {
    std::unique_ptr<Texture> visibility;
    std::unique_ptr<TextureView> visibilityView;
    std::unique_ptr<Texture> depth;
    std::unique_ptr<TextureView> depthView;
};

struct GPUDrivenPreviewFrameSlotResources {
    std::unique_ptr<Buffer> paramsBuffer;
    Buffer* instanceVisibilityBuffer = nullptr;
    Buffer* visibleInstanceIdsBuffer = nullptr;
    Buffer* visibleInstanceCounterBuffer = nullptr;
    std::array<Buffer*, 2> visibleMeshletBuffers{};
    std::array<Buffer*, 2> indirectBuffers{};
    BindlessHandle paramsHandle;
    BindlessHandle instanceVisibilityHandle;
    BindlessHandle visibleInstanceIdsHandle;
    BindlessHandle visibleInstanceCounterHandle;
    std::array<BindlessHandle, 2> visibleMeshletHandles;
    std::array<BindlessHandle, 2> indirectHandles;
};

struct GPUDrivenPreviewFrameSlotBindings {
    BindlessHandle paramsHandle;
    BindlessHandle instanceVisibilityHandle;
    BindlessHandle visibleInstanceIdsHandle;
    BindlessHandle visibleInstanceCounterHandle;
    std::array<BindlessHandle, 2> visibleMeshletHandles;
    std::array<BindlessHandle, 2> indirectHandles;
};

struct GPUDrivenStreamDeferredBindings {
    uint32_t pageBuffer = 0;
    uint32_t activeGroupBuffer = 0;
    uint32_t pageTableBuffer = 0;
    uint32_t activeHeaderBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t visibleClusterBuffer = 0;
    uint32_t visibleRecordBase = 0;
    uint32_t visibleRecordCapacity = 0;
};

static_assert(sizeof(GPUDrivenStreamDeferredBindings) == 32);

enum GPUDrivenStreamDeferredResourceIndex : uint32_t {
    GPUDrivenStreamDeferredPage,
    GPUDrivenStreamDeferredActiveGroup,
    GPUDrivenStreamDeferredPageTable,
    GPUDrivenStreamDeferredActiveHeader,
    GPUDrivenStreamDeferredParams,
    GPUDrivenStreamDeferredVisibleCluster,
    GPUDrivenStreamDeferredResourceCount,
};

struct GPUDrivenPreviewBindingBundle {
    std::unique_ptr<Buffer> materialTextureRemapBuffer;
    std::unique_ptr<Buffer> streamDeferredBindingsBuffer;
    std::unique_ptr<Buffer> streamOwnerMaskBuffer;
    std::unique_ptr<BindlessHeap> heap;
    GPUSceneConsumerBindings gpuSceneBindings;
    BindlessHandle materialTextureRemapHandle;
    std::vector<GPUDrivenPreviewFrameSlotBindings> frameSlots;
    std::array<BindlessHandle, 2> hzbHandles;
    BindlessHandle deferredColorHandle;
    BindlessHandle depthImageHandle;
    BindlessHandle visibilityImageHandle;
    BindlessHandle cullingDepthImageHandle;
    std::vector<BindlessHandle> materialTextureHandles;
    BindlessHandle environmentTextureHandle;
    BindlessHandle environmentSHBufferHandle;
    std::array<BindlessHandle, kGPUDrivenOpenPBRLut2DCount> openPBRLut2DHandles;
    std::array<BindlessHandle, kGPUDrivenOpenPBRLut3DCount> openPBRLut3DHandles;
    std::array<BindlessHandle, GPUDrivenStreamDeferredResourceCount>
        streamDeferredResourceHandles;
    BindlessHandle streamDeferredBindingsHandle;
    BindlessHandle streamOwnerMaskHandle;
};

struct GPUDrivenPreviewRetiredViewResources {
    std::unique_ptr<Buffer> deferredColorBuffer;
    GPUDrivenPreviewCullingTargets cullingTargets;
    std::unique_ptr<Buffer> materialTextureRemapBuffer;
    std::unique_ptr<Buffer> streamDeferredBindingsBuffer;
    std::unique_ptr<Buffer> streamOwnerMaskBuffer;
    std::unique_ptr<BindlessHeap> bindlessHeap;
};

class GPUDrivenPreviewPass final : public UnsafePass {
public:
    ~GPUDrivenPreviewPass() override
    {
        releaseGPUSceneSourceLease();
        if (gpuSceneSubsystem_ != nullptr && gpuSceneView_.valid()) {
            gpuSceneSubsystem_->destroyView(gpuSceneView_);
        }
    }

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{
            EnvironmentLightingSubsystem::kSubsystemId,
            GPUSceneSubsystem::kSubsystemId,
        };
        return required;
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "OpenPBR deferred shading and meshlet visualization")
            .format = Format::Rgba8Unorm;
        RenderGraphField& visibility = reflection.addTextureOutput(
            "visibility",
            "Mesh shader visibility buffer");
        visibility.colorWrite();
        visibility.format = Format::R32Uint;
        visibility.usage = visibility.usage | TextureUsageBits::Sampled;
        RenderGraphField& depth = reflection.addTextureOutput(
            "depth",
            "Visibility pass depth and HZB source");
        depth.depthStencilWrite();
        depth.usage = depth.usage | TextureUsageBits::Sampled;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeEnumSetting(
                "mode",
                "Mode",
                "meshlet",
                {{"Shaded", "shaded"}, {"Base Color", "baseColor"}, {"Meshlet", "meshlet"}, {"Primitive", "primitive"}, {"LOD Group", "lod"}}),
            runtimeIntSetting("lodLevel", "LOD Level", 0, 0, 31),
            runtimeBoolSetting("instanceFrustumCull", "Instance Frustum Cull", true),
            runtimeBoolSetting("instanceHzbCull", "Instance HZB Cull", true),
            runtimeBoolSetting("meshletFrustumCull", "Meshlet Sphere / Frustum Cull", true),
            runtimeBoolSetting("meshletNormalConeCull", "Meshlet Normal Cone Cull", true),
            runtimeBoolSetting("freezeCullingCamera", "Freeze Culling Camera", false),
        };
        appendCameraRuntimeSettings(
            settings,
            std::array<float, 3>{0.0f, 2.0f, 8.0f},
            std::array<float, 3>{0.0f, 1.0f, 0.0f},
            60.0f);
        return settings;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        // Slang 2026.1.2 lowers fragment SV_PrimitiveID to the SPIR-V Geometry
        // capability even when it is sourced by a mesh shader primitive output.
        if (!context.device->capabilities().taskShader ||
            !context.device->capabilities().meshShader ||
            !context.device->capabilities().geometryShader ||
            !context.device->capabilities().bindlessDescriptorHeap) {
            log = "GPUDrivenPreviewPass requires taskShader, meshShader, geometryShader, and bindlessDescriptorHeap capabilities";
            return makeError(Error::Unsupported);
        }
        GPUSceneSubsystem* gpuSceneSubsystem = context.subsystem<GPUSceneSubsystem>();
        if (gpuSceneSubsystem == nullptr) {
            log = "GPUDrivenPreviewPass requires GPUSceneSubsystem";
            return makeError(Error::InvalidArgument);
        }
        if (gpuSceneSubsystem_ != gpuSceneSubsystem) {
            releaseGPUSceneSourceLease();
            if (gpuSceneSubsystem_ != nullptr && gpuSceneView_.valid()) {
                gpuSceneSubsystem_->destroyView(gpuSceneView_);
            }
            gpuSceneSubsystem_ = gpuSceneSubsystem;
            gpuSceneView_ = {};
            gpuSceneSource_ = nullptr;
            compiledScene_ = nullptr;
        }
        const uint32_t requestedFrameSlotCount = std::max(
            gpuSceneSubsystem_->frameSlotCount(),
            1u);
        if (gpuSceneView_.valid() &&
            frameSlotCount_ != 0 &&
            frameSlotCount_ != requestedFrameSlotCount) {
            gpuSceneSubsystem_->destroyView(gpuSceneView_);
            gpuSceneView_ = {};
        }
        if (!gpuSceneView_.valid()) {
            gpuSceneView_ = gpuSceneSubsystem_->createView(GPUSceneViewDesc{
                .frameSlotCount = requestedFrameSlotCount,
            });
            if (!gpuSceneView_.valid()) {
                log = "GPUDrivenPreviewPass failed to allocate a GPUScene View";
                return makeError(Error::Failure);
            }
        }
        device_ = context.device;
        const scene::Scene* runtimeScene = runtimeSceneForPath(
            context.runtimeScene,
            scenePathFromProperties(properties()));
        if (runtimeScene == nullptr && context.sceneResourceManager != nullptr) {
            Result sceneResult = context.sceneResourceManager->resolveScene(
                properties(), context.runtimeScene, runtimeScene, log);
            if (!sceneResult) {
                return sceneResult;
            }
        }
        const bool requestedStreamEnabled = previewStreamEnabled(properties());
        PreviewStreamSourceSelection requestedStreamSource;
        std::filesystem::path requestedStreamAssetPath;
        if (requestedStreamEnabled) {
            if (!resolvePreviewStreamSource(
                    properties(),
                    runtimeScene,
                    scenePathFromProperties(properties()),
                    requestedStreamSource,
                    log)) {
                return makeError(Error::InvalidArgument);
            }
            requestedStreamAssetPath = previewStreamAssetPath(
                properties(),
                requestedStreamSource.sourcePath);
        }
        if (gpuSceneSource_ != runtimeScene) {
            releaseGPUSceneSourceLease();
            gpuSceneSource_ = runtimeScene;
            if (gpuSceneSource_ != nullptr) {
                Result leaseResult = gpuSceneSubsystem_->acquireSourceOverride(
                    gpuSceneSource_,
                    gpuSceneSourceToken_,
                    log);
                if (!leaseResult) {
                    gpuSceneSource_ = nullptr;
                    return leaseResult;
                }
            }
        }
        const uint64_t runtimeResourceIdentity = runtimeScene != nullptr
            ? runtimeScene->resourceIdentity()
            : 0;
        const uint64_t runtimeRevision = runtimeScene != nullptr ? runtimeScene->transformRevision() : 0;
        const uint64_t runtimeLifetimeRevision = runtimeScene != nullptr
            ? runtimeScene->sceneGraph().lifetimeRevision()
            : 0;
        const uint64_t runtimeStructuralRevision = runtimeScene != nullptr
            ? runtimeScene->sceneGraph().structuralRevision()
            : 0;
        const uint64_t runtimeContentRevision = runtimeScene != nullptr
            ? runtimeScene->contentRevision()
            : 0;
        if (visibilityPipelines_[0] != nullptr &&
            drawTaskCount_ > 0 &&
            compiledScene_ == runtimeScene &&
            sceneResourceIdentity_ == runtimeResourceIdentity &&
            sceneLifetimeRevision_ == runtimeLifetimeRevision &&
            sceneStructuralRevision_ == runtimeStructuralRevision &&
            sceneContentRevision_ == runtimeContentRevision &&
            frameWidth_ == context.width &&
            frameHeight_ == context.height &&
            frameSlotResources_.size() == requestedFrameSlotCount &&
            streamEnabled_ == requestedStreamEnabled &&
            (!requestedStreamEnabled ||
                (compiledStreamAssetPath_ == requestedStreamAssetPath &&
                    compiledStreamSourceId_ == requestedStreamSource.sourceId &&
                    compiledStreamSourcePath_ == requestedStreamSource.sourcePath))) {
            return {};
        }

        std::vector<GPUDrivenPreviewGpuVertex> vertices;
        std::vector<GPUDrivenPreviewGpuMeshlet> meshlets;
        std::vector<GPUDrivenPreviewGpuMeshletDraw> meshletDraws;
        std::vector<uint32_t> meshletVertices;
        std::vector<uint32_t> meshletTriangles;
        std::vector<SceneGpuTransform> transforms;
        std::vector<GPUDrivenPreviewGpuInstance> instances;
        std::vector<uint32_t> instanceRenderNodeIndices;
        std::vector<GPUDrivenPreviewMeshletRange> lodLevelRanges;
        GPUDrivenPreviewMeshletRange baseMeshletRange;
        auto compileStageBegin = GPUDrivenCompileClock::now();
        if (!loadMeshletScene(
                properties(),
                runtimeScene,
                vertices,
                meshlets,
                meshletDraws,
                meshletVertices,
                meshletTriangles,
                transforms,
                instances,
                instanceRenderNodeIndices,
                drawBounds_,
                baseMeshletRange,
                lodLevelRanges,
                log)) {
            return makeError(Error::Failure);
        }
        spdlog::info(
            "[GPUDrivenPreviewPass] CPU raster plan vertices={} uniqueMeshlets={} meshletDraws={} instances={}; GPU payload is owned by GPUScene",
            vertices.size(),
            meshlets.size(),
            meshletDraws.size(),
            instances.size());
        logGPUDrivenCompileStage("meshlet scene", compileStageBegin);
        if (runtimeScene == nullptr) {
            log = "GPUDrivenPreviewPass shading requires a runtime scene";
            return makeError(Error::InvalidArgument);
        }
        if (requestedStreamEnabled) {
            resetStreamIntegration();
            Result streamResult = streamRuntime_.initialize(
                *context.device,
                previewStreamRuntimeDesc(
                    properties(),
                    requestedStreamSource.sourcePath),
                log);
            if (!streamResult) {
                log = "GPUDrivenPreviewPass stream integration failed: " + log;
                return streamResult;
            }
            streamEnabled_ = true;
            compiledStreamAssetPath_ = requestedStreamAssetPath;
            compiledStreamSourceId_ = requestedStreamSource.sourceId;
            compiledStreamSourcePath_ = requestedStreamSource.sourcePath;
            streamResult = allocateStreamRasterBindings(log);
            if (!streamResult) {
                return streamResult;
            }
        } else {
            resetStreamIntegration();
        }
        std::vector<GPUDrivenPreviewGpuMaterial> materials;
        compileStageBegin = GPUDrivenCompileClock::now();
        Result result = prepareShadingResources(
            *context.device,
            *runtimeScene,
            properties(),
            materials,
            log);
        if (!result) {
            return result;
        }
        logGPUDrivenCompileStage("shading resources", compileStageBegin);
        materialCount_ = static_cast<uint32_t>(materials.size());
        materialTextureCount_ = materialCount_ * kGPUSceneMaterialTextureSlotCount;

        baseMeshletRange_ = baseMeshletRange;
        lodLevelRanges_ = std::move(lodLevelRanges);
        drawTaskCount_ = maxMeshletRangeCount(baseMeshletRange_, lodLevelRanges_);
        instanceCount_ = static_cast<uint32_t>(instances.size());
        if (drawTaskCount_ == 0 || instanceCount_ == 0) {
            log = "GPUDrivenPreviewPass found no drawable meshlet instances";
            return makeError(Error::Failure);
        }
        residentRecordCapacity_ = static_cast<uint32_t>(meshletDraws.size());
        streamOwnerMask_.assign(std::max(instanceCount_, 1u), 0u);
        if (streamEnabled_ && !visibilityRecordRangeFitsId(
                residentRecordCapacity_,
                streamRuntime_.visibleClusterCapacity())) {
            log = "GPUDrivenPreviewPass resident + stream records exceed the visibility ID range";
            return makeError(Error::InvalidArgument);
        }

        frameWidth_ = std::max(context.width, 1u);
        frameHeight_ = std::max(context.height, 1u);
        hzbMipCount_ = computeHzbMipCount(frameWidth_, frameHeight_);
        hzbElementCount_ = computeHzbElementCount(frameWidth_, frameHeight_, hzbMipCount_);
        frameIndex_ = 0;
        invalidateHzbHistory();
        previousCameraValid_ = false;
        frameBuffersInitialized_ = false;
        cullingTargetsInitialized_ = false;
        freezeCullingCamera_ = false;
        frozenCullingCameraValid_ = false;
        frameSlotCount_ = requestedFrameSlotCount;
        activeFrameSlot_ = 0;
        frameSlotResources_.clear();
        frameSlotResources_.resize(frameSlotCount_);

        GPUDrivenPreviewGpuParams params;
        const EnvironmentSettings initialEnvironment = context.world() != nullptr
            ? context.world()->environment()
            : EnvironmentSettings{};
        buildParams(
            context.width,
            context.height,
            properties(),
            drawBounds_,
            baseMeshletRange_,
            lodLevelRanges_,
            instanceCount_,
            hzbMipCount_,
            frameIndex_,
            false,
            nullptr,
            materialTextureCount_,
            materialCount_,
            initialEnvironment,
            false,
            params);

        compileStageBegin = GPUDrivenCompileClock::now();
        for (uint32_t frameSlot = 0; frameSlot < frameSlotCount_; ++frameSlot) {
            result = uploadStorageBuffer(
                *context.device,
                &params,
                sizeof(params),
                frameSlotResources_[frameSlot].paramsBuffer,
                log,
                "GPUDrivenPreviewPass frame-slot params");
            if (!result) {
                return result;
            }
        }
        result = ensureGPUSceneViewResources(frameWidth_, frameHeight_, log);
        if (!result) {
            return result;
        }
        result = createDeviceStorageBuffer(
            *context.device,
            static_cast<uint64_t>(frameWidth_) * frameHeight_ * sizeof(uint32_t),
            BufferUsageBits::Storage,
            deferredColorBuffer_,
            log,
            "deferred color");
        if (!result) {
            return result;
        }
        result = createCullingTargets(
            *context.device,
            frameWidth_,
            frameHeight_,
            cullingTargets_,
            log);
        if (!result) {
            return result;
        }

        GPUDrivenPreviewBindingBundle initialBindings;
        result = createBindingBundle(
            *deferredColorBuffer_,
            *cullingTargets_.depthView,
            false,
            initialBindings,
            log);
        if (!result) {
            return result;
        }
        installBindingBundle(std::move(initialBindings));
        logGPUDrivenCompileStage("GPU resources and descriptors", compileStageBegin);
        compileStageBegin = GPUDrivenCompileClock::now();
        result = createPipelines(*context.device, log);
        if (!result) {
            return result;
        }
        logGPUDrivenCompileStage("shader and compute pipelines", compileStageBegin);
        compileStageBegin = GPUDrivenCompileClock::now();
        for (uint32_t bucketIndex = 0;
             bucketIndex < kGPUDrivenPreviewDrawBucketCount;
             ++bucketIndex) {
            const bool masked = bucketIndex >= 2u;
            const bool doubleSided = (bucketIndex & 1u) != 0u;
            result = context.device->createGraphicsPipeline(
                GraphicsPipelineDesc{
                    .taskShader = amplificationShader_.get(),
                    .meshShader = meshShader_.get(),
                    .fragmentShader = masked ? maskedFragmentShader_.get() : fragmentShader_.get(),
                    .colorFormat = Format::R32Uint,
                    .depthStencilFormat = Format::D32Sfloat,
                    .rasterization = RasterizationState{
                        .cullMode = doubleSided ? CullMode::None : CullMode::Back,
                        // gpuDrivenPreviewMeshMain flips clip-space Y while the
                        // positive-height Vulkan viewport flips winding again.
                        .frontFace = FrontFace::CounterClockwise,
                    },
                    .depthStencil = DepthStencilState{
                        .depthTestEnable = true,
                        .depthWriteEnable = true,
                        .depthCompareOp = depthCompareOp(kDefaultReversedZ),
                    },
                    .usesBindlessHeap = true,
                    .pipelineCache = pipelineCache_.get(),
                },
                visibilityPipelines_[bucketIndex]);
            if (!result || visibilityPipelines_[bucketIndex] == nullptr) {
                log += resultMessage(
                    "createGraphicsPipeline(GPUDrivenPreviewPass visibility bucket)",
                    result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
        }
        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .vertexShader = compositeVertexShader_.get(),
                .fragmentShader = compositeFragmentShader_.get(),
                .colorFormat = Format::Rgba8Unorm,
                .topology = PrimitiveTopology::TriangleList,
                .usesBindlessHeap = true,
                .pipelineCache = pipelineCache_.get(),
            },
            compositePipeline_);
        if (!result || compositePipeline_ == nullptr) {
            log += resultMessage("createGraphicsPipeline(GPUDrivenPreviewPass composite)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        logGPUDrivenCompileStage("graphics pipelines", compileStageBegin);
        if (pipelineCache_ != nullptr) {
            const Result saveResult = pipelineCache_->save();
            const PipelineCacheStats stats = pipelineCache_->stats();
            spdlog::info(
                "[GPUDrivenPreviewPass] PSO cache status={} hits={} misses={} stored={} bytes={}",
                pipelineCacheLoadStatusName(stats.loadStatus),
                stats.hitCount,
                stats.missCount,
                stats.storedPsoCount,
                stats.backendDataSize);
            if (!saveResult) {
                log += "Warning: GPUDrivenPreviewPass failed to save PSO cache\n";
            }
        }

        sceneResourceIdentity_ = runtimeResourceIdentity;
        sceneRevision_ = runtimeRevision;
        sceneVisibilityRevision_ = runtimeScene != nullptr ? runtimeScene->visibilityRevision() : 0;
        sceneLifetimeRevision_ = runtimeLifetimeRevision;
        sceneStructuralRevision_ = runtimeStructuralRevision;
        sceneContentRevision_ = runtimeContentRevision;
        compiledScene_ = runtimeScene;
        observedHistoryInvalidationRevision_ = 0;
        previousParams_ = params;
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        Result syncResult = syncRuntimeGeometry(context.runtimeScene());
        if (!syncResult) {
            return syncResult;
        }
        EnvironmentLightingSubsystem* environmentSubsystem =
            context.subsystem<EnvironmentLightingSubsystem>();
        GPUSceneSubsystem* gpuSceneSubsystem = context.subsystem<GPUSceneSubsystem>();
        if (environmentSubsystem == nullptr || gpuSceneSubsystem == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (gpuSceneSubsystem != gpuSceneSubsystem_ || !gpuSceneView_.valid()) {
            return makeError(Error::InvalidArgument);
        }
        const EnvironmentLightingSnapshot& environment = environmentSubsystem->snapshot();
        if (!environment.valid()) {
            return {};
        }
        TextureHandle color = context.outputTexture("color");
        TextureHandle visibility = context.outputTexture("visibility");
        TextureHandle depth = context.outputTexture("depth");
        if (!color.valid() ||
            !visibility.valid() ||
            !depth.valid() ||
            bindlessHeap_ == nullptr ||
            visibilityPipelines_[0] == nullptr ||
            compositePipeline_ == nullptr ||
            cullingTargets_.visibilityView == nullptr ||
            cullingTargets_.depthView == nullptr ||
            drawTaskCount_ == 0 ||
            (streamEnabled_ &&
                (context.streamer() == nullptr ||
                    !streamRuntime_.ready() ||
                    streamVisibilityPipeline_ == nullptr ||
                    streamCullResetPipeline_ == nullptr ||
                    streamInstanceCullPipeline_ == nullptr))) {
            return makeError(Error::InvalidArgument);
        }

        std::string gpuSceneLog;
        Result result = syncGPUSceneRasterState(
            *gpuSceneSubsystem,
            gpuSceneLog);
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", gpuSceneLog);
            return result;
        }
        result = ensureFrameResources(
            context.width(),
            context.height(),
            context.subsystems());
        if (!result) {
            return result;
        }
        const uint32_t frameSlot = gpuSceneSubsystem->currentFrameSlot();
        if (frameSlot >= frameSlotResources_.size()) {
            return makeError(Error::InvalidArgument);
        }
        activeFrameSlot_ = frameSlot;
        MeshletStreamFrameDesc streamFrame;
        if (streamEnabled_) {
            gpuSceneLog.clear();
            result = syncStreamRuntimeScene(
                context.runtimeScene(),
                *gpuSceneSubsystem,
                gpuSceneLog);
            if (!result) {
                spdlog::error("[GPUDrivenPreviewPass] {}", gpuSceneLog);
                return result;
            }
            streamFrame = streamFrameDesc(context);
            result = streamRuntime_.cmdBeginFrame(
                context.commandBuffer(),
                *context.streamer(),
                streamFrame);
            if (!result) {
                return result;
            }
            // Both producer families must select the same HZB history index.
            frameIndex_ = streamRuntime_.frameIndex();
        }
        bool cameraCut = false;
        if (HistoryResourceManager* historyResources = context.historyResources()) {
            const uint64_t invalidationRevision =
                historyResources->invalidationRevision();
            cameraCut = observedHistoryInvalidationRevision_ != 0 &&
                observedHistoryInvalidationRevision_ != invalidationRevision;
            observedHistoryInvalidationRevision_ = invalidationRevision;
        }
        result = prepareGPUSceneView(*gpuSceneSubsystem, cameraCut);
        if (!result) {
            return result;
        }
        if (context.subsystems() == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        gpuSceneLog.clear();
        result = ensureGPUSceneBindings(
            *gpuSceneSubsystem,
            *context.subsystems(),
            gpuSceneLog);
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", gpuSceneLog);
            return result;
        }
        result = updateParamsBuffer(
            context.width(),
            context.height(),
            context.properties(),
            environment.settings,
            environment.mapAvailable);
        if (!result) {
            return result;
        }
        if (!environmentBindingValid_ ||
            environment.resourceRevision != environmentResourceRevision_) {
            result = bindlessHeap_->writeSampledImage(
                environmentTextureHandle_,
                *environment.radianceView,
                ResourceState::ShaderRead);
            if (!result) {
                return result;
            }
            result = bindlessHeap_->writeStorageBuffer(
                environmentSHBufferHandle_,
                *environment.sphericalHarmonicsBuffer);
            if (!result) {
                return result;
            }
            environmentResourceRevision_ = environment.resourceRevision;
            environmentBindingValid_ = true;
        }

        result = bindlessHeap_->writeSampledImage(
            depthImageHandle_,
            *depth.view(),
            ResourceState::ShaderRead);
        if (!result) {
            return result;
        }
        result = bindlessHeap_->writeSampledImage(
            visibilityImageHandle_,
            *visibility.view(),
            ResourceState::ShaderRead);
        if (!result) {
            return result;
        }
        if (streamEnabled_) {
            result = bindStreamViewResources(
                *gpuSceneSubsystem,
                visibility,
                depth);
            if (!result) {
                return result;
            }
            result = streamRuntime_.cmdPreTraversal(
                context.commandBuffer(),
                streamFrame);
            if (!result) {
                return result;
            }
        }

        CommandBuffer& commandBuffer = context.commandBuffer();
        result = uploadShadingTextures(commandBuffer);
        if (!result) {
            return result;
        }
        commandBuffer.bindBindlessHeap(*bindlessHeap_);
        result = initializeInternalBuffers(commandBuffer);
        if (!result) {
            return result;
        }

        result = dispatchCulling(commandBuffer, 0);
        if (!result) {
            return result;
        }
        if (streamEnabled_) {
            result = dispatchStreamCulling(
                commandBuffer,
                GPUSceneCullPhase::Early);
            if (!result) {
                return result;
            }
        }
        if (freezeCullingCamera_) {
            drawVisibility(
                commandBuffer,
                *cullingTargets_.visibilityView,
                *cullingTargets_.depthView,
                0,
                LoadOp::Clear,
                true);
            transitionTexture(
                commandBuffer,
                *cullingTargets_.depth,
                ResourceState::DepthStencilAttachment,
                ResourceState::ShaderRead);
            result = buildHzb(commandBuffer);
            if (!result) {
                return result;
            }
            transitionTexture(
                commandBuffer,
                *cullingTargets_.depth,
                ResourceState::ShaderRead,
                ResourceState::DepthStencilAttachment);
            drawVisibility(commandBuffer, *visibility.view(), *depth.view(), 0, LoadOp::Clear);
            if (streamEnabled_) {
                result = drawStreamVisibility(
                    commandBuffer,
                    *visibility.view(),
                    *depth.view(),
                    GPUSceneCullPhase::Early,
                    LoadOp::Load);
                if (!result) {
                    return result;
                }
            }
        } else {
            drawVisibility(commandBuffer, *visibility.view(), *depth.view(), 0, LoadOp::Clear);
            if (streamEnabled_) {
                result = drawStreamVisibility(
                    commandBuffer,
                    *visibility.view(),
                    *depth.view(),
                    GPUSceneCullPhase::Early,
                    LoadOp::Load);
                if (!result) {
                    return result;
                }
            }
            transitionTexture(commandBuffer, *depth.texture(), ResourceState::DepthStencilAttachment, ResourceState::ShaderRead);
            result = buildHzb(commandBuffer);
            if (!result) {
                return result;
            }
            transitionTexture(commandBuffer, *depth.texture(), ResourceState::ShaderRead, ResourceState::DepthStencilAttachment);
        }

        result = dispatchCulling(commandBuffer, 1);
        if (!result) {
            return result;
        }
        if (streamEnabled_) {
            result = dispatchStreamCulling(
                commandBuffer,
                GPUSceneCullPhase::Late);
            if (!result) {
                return result;
            }
        }
        if (freezeCullingCamera_) {
            drawVisibility(
                commandBuffer,
                *cullingTargets_.visibilityView,
                *cullingTargets_.depthView,
                1,
                LoadOp::Load,
                true);
            transitionTexture(
                commandBuffer,
                *cullingTargets_.depth,
                ResourceState::DepthStencilAttachment,
                ResourceState::ShaderRead);
            result = buildHzb(commandBuffer);
            if (!result) {
                return result;
            }
            transitionTexture(
                commandBuffer,
                *cullingTargets_.depth,
                ResourceState::ShaderRead,
                ResourceState::DepthStencilAttachment);
            drawVisibility(commandBuffer, *visibility.view(), *depth.view(), 1, LoadOp::Load);
            if (streamEnabled_) {
                result = drawStreamVisibility(
                    commandBuffer,
                    *visibility.view(),
                    *depth.view(),
                    GPUSceneCullPhase::Late,
                    LoadOp::Load);
                if (!result) {
                    return result;
                }
            }
        } else {
            drawVisibility(commandBuffer, *visibility.view(), *depth.view(), 1, LoadOp::Load);
            if (streamEnabled_) {
                result = drawStreamVisibility(
                    commandBuffer,
                    *visibility.view(),
                    *depth.view(),
                    GPUSceneCullPhase::Late,
                    LoadOp::Load);
                if (!result) {
                    return result;
                }
            }
        }

        transitionTexture(commandBuffer, *visibility.texture(), ResourceState::ColorAttachment, ResourceState::ShaderRead);
        transitionTexture(commandBuffer, *depth.texture(), ResourceState::DepthStencilAttachment, ResourceState::ShaderRead);
        if (!freezeCullingCamera_) {
            result = buildHzb(commandBuffer);
            if (!result) {
                return result;
            }
        }
        if (streamEnabled_) {
            result = streamRuntime_.cmdPrepareDeferred(commandBuffer);
            if (!result) {
                return result;
            }
            commandBuffer.bindBindlessHeap(*bindlessHeap_);
        }
        dispatchDeferred(commandBuffer);
        barrierBuffer(commandBuffer, *deferredColorBuffer_, ResourceState::General, ResourceState::General);
        transitionTexture(commandBuffer, *visibility.texture(), ResourceState::ShaderRead, ResourceState::ColorAttachment);
        transitionTexture(commandBuffer, *depth.texture(), ResourceState::ShaderRead, ResourceState::DepthStencilAttachment);
        drawComposite(commandBuffer, color);

        hzbValid_ = true;
        if (!gpuSceneSubsystem->markViewHzbValid(
                gpuSceneView_,
                activeFrameSlot_,
                true)) {
            return makeError(Error::Failure);
        }
        if (streamEnabled_) {
            result = streamRuntime_.cmdPostTraversal(commandBuffer);
            if (!result) {
                return result;
            }
            result = streamRuntime_.cmdEndFrame(commandBuffer);
            if (!result) {
                return result;
            }
        }
        ++frameIndex_;
        return {};
    }

private:
    void resetStreamIntegration()
    {
        streamRuntime_.reset();
        streamMeshShader_.reset();
        streamFragmentShader_.reset();
        streamCullResetShader_.reset();
        streamInstanceCullShader_.reset();
        streamVisibilityPipeline_.reset();
        streamCullResetPipeline_.reset();
        streamInstanceCullPipeline_.reset();
        streamVisibilityImageHandle_ = {};
        streamDepthImageHandle_ = {};
        streamInstanceVisibilityHandle_ = {};
        streamVisibleInstanceIdsHandle_ = {};
        streamVisibleInstanceCounterHandle_ = {};
        streamHzbHandles_ = {};
        streamDeferredResourceHandles_ = {};
        streamDeferredBindingsHandle_ = {};
        streamOwnerMaskHandle_ = {};
        streamDeferredBindingsBuffer_.reset();
        streamOwnerMaskBuffer_.reset();
        streamEnabled_ = false;
        compiledStreamAssetPath_.clear();
        compiledStreamSourceId_.clear();
        compiledStreamSourcePath_.clear();
        streamOwnerMask_.clear();
        streamMappedInstanceCount_ = 0;
    }

    Result allocateStreamRasterBindings(std::string& log)
    {
        BindlessHeap* heap = streamRuntime_.bindlessHeap();
        if (!streamEnabled_ || heap == nullptr) {
            log = "GPUDrivenPreviewPass stream runtime has no bindless heap";
            return makeError(Error::InvalidArgument);
        }
        auto allocateBuffer = [&](BindlessHandle& handle,
                                  const char* label) -> Result {
            Result result = heap->allocateBuffer(handle);
            if (!result || !handle.valid()) {
                log += resultMessage(
                    std::string("allocateBuffer(GPUDrivenPreviewPass stream ") +
                        label + ")",
                    result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
            return {};
        };
        auto allocateImage = [&](BindlessHandle& handle,
                                 const char* label) -> Result {
            Result result = heap->allocateSampledImage(handle);
            if (!result || !handle.valid()) {
                log += resultMessage(
                    std::string("allocateSampledImage(GPUDrivenPreviewPass stream ") +
                        label + ")",
                    result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
            return {};
        };

        Result result = allocateImage(streamVisibilityImageHandle_, "visibility");
        if (result) {
            result = allocateImage(streamDepthImageHandle_, "depth");
        }
        if (result) {
            result = allocateBuffer(
                streamInstanceVisibilityHandle_,
                "instance visibility");
        }
        if (result) {
            result = allocateBuffer(
                streamVisibleInstanceIdsHandle_,
                "visible instance IDs");
        }
        if (result) {
            result = allocateBuffer(
                streamVisibleInstanceCounterHandle_,
                "visible instance counter");
        }
        for (uint32_t historyIndex = 0;
             historyIndex < streamHzbHandles_.size() && result;
             ++historyIndex) {
            result = allocateBuffer(
                streamHzbHandles_[historyIndex],
                "HZB history");
        }
        return result;
    }

    void invalidateHzbHistory()
    {
        hzbValid_ = false;
        ++hzbHistoryEpoch_;
        if (hzbHistoryEpoch_ == 0) {
            hzbHistoryEpoch_ = 1;
        }
    }

    void releaseGPUSceneSourceLease()
    {
        if (gpuSceneSubsystem_ != nullptr && gpuSceneSourceToken_.valid()) {
            (void)gpuSceneSubsystem_->releaseSourceOverride(gpuSceneSourceToken_);
        }
        gpuSceneSourceToken_ = {};
        gpuSceneSource_ = nullptr;
    }

    static uint32_t divideRoundUp(uint32_t value, uint32_t divisor)
    {
        return (value + divisor - 1u) / divisor;
    }

    static uint32_t computeHzbMipCount(uint32_t width, uint32_t height)
    {
        uint32_t mipCount = 1;
        while (width > 1 || height > 1) {
            width = std::max(1u, (width + 1u) / 2u);
            height = std::max(1u, (height + 1u) / 2u);
            ++mipCount;
        }
        return mipCount;
    }

    static uint64_t computeHzbElementCount(uint32_t width, uint32_t height, uint32_t mipCount)
    {
        uint64_t elementCount = 0;
        for (uint32_t mipLevel = 0; mipLevel < mipCount; ++mipLevel) {
            elementCount += static_cast<uint64_t>(width) * height;
            width = std::max(1u, (width + 1u) / 2u);
            height = std::max(1u, (height + 1u) / 2u);
        }
        return elementCount;
    }

    static Result createDeviceStorageBuffer(
        Device& device,
        uint64_t byteSize,
        BufferUsageBits usage,
        std::unique_ptr<Buffer>& outBuffer,
        std::string& log,
        std::string_view label)
    {
        if (byteSize == 0) {
            log = std::string("GPUDrivenPreviewPass ") + std::string(label) + " buffer size is zero";
            return makeError(Error::InvalidArgument);
        }
        Result result = device.createBuffer(
            BufferDesc{
                .size = byteSize,
                .structureStride = 0,
                .usage = usage,
                .memoryLocation = MemoryLocation::Device,
            },
            outBuffer);
        if (!result || outBuffer == nullptr) {
            log += resultMessage(std::string("createBuffer(GPUDrivenPreviewPass ") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    static Result createCullingTargets(
        Device& device,
        uint32_t width,
        uint32_t height,
        GPUDrivenPreviewCullingTargets& outTargets,
        std::string& log)
    {
        GPUDrivenPreviewCullingTargets targets;
        Result result = device.createTexture(
            TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::ColorAttachment,
                .format = Format::R32Uint,
                .width = std::max(width, 1u),
                .height = std::max(height, 1u),
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            },
            targets.visibility);
        if (!result || targets.visibility == nullptr) {
            log += resultMessage("createTexture(GPUDrivenPreviewPass culling visibility)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createTextureView(
            *targets.visibility,
            TextureViewDesc{.format = Format::R32Uint},
            targets.visibilityView);
        if (!result || targets.visibilityView == nullptr) {
            log += resultMessage("createTextureView(GPUDrivenPreviewPass culling visibility)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = device.createTexture(
            TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::DepthStencilAttachment | TextureUsageBits::Sampled,
                .format = Format::D32Sfloat,
                .width = std::max(width, 1u),
                .height = std::max(height, 1u),
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            },
            targets.depth);
        if (!result || targets.depth == nullptr) {
            log += resultMessage("createTexture(GPUDrivenPreviewPass culling depth)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createTextureView(
            *targets.depth,
            TextureViewDesc{.format = Format::D32Sfloat},
            targets.depthView);
        if (!result || targets.depthView == nullptr) {
            log += resultMessage("createTextureView(GPUDrivenPreviewPass culling depth)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        outTargets = std::move(targets);
        return {};
    }

    static Result createShader(
        Device& device,
        const char* moduleName,
        const char* entryPoint,
        bool meshShadingShader,
        std::unique_ptr<ShaderModule>& outShader,
        std::string& log,
        const SlangMacroDefine* macroDefines = nullptr,
        uint32_t macroDefineCount = 0)
    {
        ShaderCompileResult compileResult;
        const char* capabilities[] = {"spvMeshShadingEXT"};
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = moduleName,
                .entryPointName = entryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .capabilities = meshShadingShader ? capabilities : nullptr,
                .capabilityCount = meshShadingShader
                    ? static_cast<uint32_t>(std::size(capabilities))
                    : 0u,
                .macroDefines = macroDefines,
                .macroDefineCount = macroDefineCount,
            },
            compileResult);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += moduleName;
            log += ".";
            log += entryPoint;
            log += ") returned ";
            log += resultToString(result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            log += '\n';
            return result;
        }
        const std::string shaderDebugName = std::string(moduleName) + "." + entryPoint;
        result = device.createShaderModule(
            ShaderModuleDesc{
                .code = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
                .debugName = shaderDebugName.c_str(),
            },
            outShader);
        if (!result) {
            log += resultMessage(
                std::string("createShaderModule(GPUDrivenPreviewPass ") + entryPoint + ")",
                result);
            log += '\n';
        }
        return result;
    }

    Result createPipelines(Device& device, std::string& log)
    {
        struct ShaderRequest {
            const char* moduleName = kGPUDrivenPreviewShaderModuleName;
            const char* entryPoint = nullptr;
            bool meshShadingShader = false;
            std::unique_ptr<ShaderModule>* shader = nullptr;
        };
        const std::array<ShaderRequest, 10> requests{
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewAmplificationEntryPoint, true, &amplificationShader_},
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewMeshEntryPoint, true, &meshShader_},
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewFragmentEntryPoint, false, &fragmentShader_},
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewMaskedFragmentEntryPoint, false, &maskedFragmentShader_},
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewResetEntryPoint, false, &resetShader_},
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewInstanceCullEntryPoint, false, &instanceCullShader_},
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewHzbEntryPoint, false, &hzbShader_},
            ShaderRequest{kGPUDrivenDeferredShaderModuleName, kGPUDrivenPreviewDeferredEntryPoint, false, &deferredShader_},
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewCompositeVertexEntryPoint, false, &compositeVertexShader_},
            ShaderRequest{kGPUDrivenPreviewShaderModuleName, kGPUDrivenPreviewCompositeFragmentEntryPoint, false, &compositeFragmentShader_},
        };
        const std::string lut2DShaderBase = std::to_string(openPBRLut2DHandles_[0].shaderIndex);
        const std::string lut3DShaderBase = std::to_string(openPBRLut3DHandles_[0].shaderIndex);
        const std::array<SlangMacroDefine, 2> openPBRDefines{
            SlangMacroDefine{"GPU_DRIVEN_OPENPBR_LUT_2D_SHADER_BASE", lut2DShaderBase.c_str()},
            SlangMacroDefine{"GPU_DRIVEN_OPENPBR_LUT_3D_SHADER_BASE", lut3DShaderBase.c_str()},
        };
        for (const ShaderRequest& request : requests) {
            const bool isDeferred = request.moduleName == kGPUDrivenDeferredShaderModuleName;
            const auto shaderCompileBegin = GPUDrivenCompileClock::now();
            Result result = createShader(
                device,
                request.moduleName,
                request.entryPoint,
                request.meshShadingShader,
                *request.shader,
                log,
                isDeferred ? openPBRDefines.data() : nullptr,
                isDeferred ? static_cast<uint32_t>(openPBRDefines.size()) : 0u);
            if (!result) {
                return result;
            }
            logGPUDrivenCompileStage(
                std::string("Slang ") + request.entryPoint,
                shaderCompileBegin);
        }
        if (streamEnabled_) {
            const std::array<ShaderRequest, 4> streamRequests{
                ShaderRequest{kMeshletStreamShaderModuleName, kMeshletStreamMeshEntryPoint, true, &streamMeshShader_},
                ShaderRequest{kMeshletStreamShaderModuleName, kMeshletStreamFragmentEntryPoint, false, &streamFragmentShader_},
                ShaderRequest{kMeshletStreamShaderModuleName, kMeshletStreamCullResetEntryPoint, false, &streamCullResetShader_},
                ShaderRequest{kMeshletStreamShaderModuleName, kMeshletStreamInstanceCullEntryPoint, false, &streamInstanceCullShader_},
            };
            for (const ShaderRequest& request : streamRequests) {
                const auto shaderCompileBegin = GPUDrivenCompileClock::now();
                Result streamResult = createShader(
                    device,
                    request.moduleName,
                    request.entryPoint,
                    request.meshShadingShader,
                    *request.shader,
                    log);
                if (!streamResult) {
                    return streamResult;
                }
                logGPUDrivenCompileStage(
                    std::string("Slang ") + request.entryPoint,
                    shaderCompileBegin);
            }
        }

        Result result = device.createPipelineCache(
            PipelineCacheDesc{.filePath = kGPUDrivenPipelineCachePath},
            pipelineCache_);
        if (!result || pipelineCache_ == nullptr) {
            log += resultMessage("createPipelineCache(GPUDrivenPreviewPass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        auto createCompute = [&](ShaderModule& shader, std::unique_ptr<ComputePipeline>& pipeline, const char* label) {
            const auto pipelineBegin = GPUDrivenCompileClock::now();
            Result result = device.createComputePipeline(
                ComputePipelineDesc{
                    .computeShader = &shader,
                    .computeEntryPoint = "main",
                    .usesBindlessHeap = true,
                    .bindlessUserPushDataSize = sizeof(GPUDrivenPreviewUserPush),
                    .pipelineCache = pipelineCache_.get(),
                },
                pipeline);
            if (!result || pipeline == nullptr) {
                log += resultMessage(std::string("createComputePipeline(GPUDrivenPreviewPass ") + label + ")", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
            logGPUDrivenCompileStage(std::string("compute pipeline ") + label, pipelineBegin);
            return result;
        };

        result = createCompute(*resetShader_, resetPipeline_, "reset");
        if (!result) {
            return result;
        }
        result = createCompute(*instanceCullShader_, instanceCullPipeline_, "instance cull");
        if (!result) {
            return result;
        }
        result = createCompute(*hzbShader_, hzbPipeline_, "HZB");
        if (!result) {
            return result;
        }
        result = createCompute(*deferredShader_, deferredPipeline_, "deferred");
        if (!result || !streamEnabled_) {
            return result;
        }

        auto createStreamCompute = [&](ShaderModule& shader,
                                       std::unique_ptr<ComputePipeline>& pipeline,
                                       const char* label) -> Result {
            Result streamResult = device.createComputePipeline(
                ComputePipelineDesc{
                    .computeShader = &shader,
                    .computeEntryPoint = "main",
                    .usesBindlessHeap = true,
                    .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
                    .pipelineCache = pipelineCache_.get(),
                },
                pipeline);
            if (!streamResult || pipeline == nullptr) {
                log += resultMessage(
                    std::string("createComputePipeline(GPUDrivenPreviewPass stream ") +
                        label + ")",
                    streamResult);
                log += '\n';
                return streamResult ? makeError(Error::Failure) : streamResult;
            }
            return {};
        };
        result = createStreamCompute(
            *streamCullResetShader_,
            streamCullResetPipeline_,
            "cull reset");
        if (result) {
            result = createStreamCompute(
                *streamInstanceCullShader_,
                streamInstanceCullPipeline_,
                "instance cull");
        }
        if (!result) {
            return result;
        }
        result = device.createGraphicsPipeline(
            GraphicsPipelineDesc{
                .meshShader = streamMeshShader_.get(),
                .fragmentShader = streamFragmentShader_.get(),
                .colorFormat = Format::R32Uint,
                .depthStencilFormat = Format::D32Sfloat,
                .rasterization = RasterizationState{
                    .cullMode = CullMode::Back,
                    .frontFace = FrontFace::CounterClockwise,
                },
                .depthStencil = DepthStencilState{
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = depthCompareOp(kDefaultReversedZ),
                },
                .usesBindlessHeap = true,
                .pipelineCache = pipelineCache_.get(),
            },
            streamVisibilityPipeline_);
        if (!result || streamVisibilityPipeline_ == nullptr) {
            log += resultMessage(
                "createGraphicsPipeline(GPUDrivenPreviewPass stream visibility)",
                result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    GPUDrivenPreviewFrameSlotResources& activeFrameResources()
    {
        return frameSlotResources_[activeFrameSlot_];
    }

    const GPUDrivenPreviewFrameSlotResources& activeFrameResources() const
    {
        return frameSlotResources_[activeFrameSlot_];
    }

    GPUSceneViewDesc gpuSceneViewResourceDesc(
        uint32_t width,
        uint32_t height) const
    {
        constexpr uint32_t visibleMeshletCapacity = 1u;
        GPUSceneViewDesc desc{
            .frameSlotCount = std::max(frameSlotCount_, 1u),
            .instanceCapacity = std::max(instanceCount_, 1u),
            .hzbWidth = std::max(width, 1u),
            .hzbHeight = std::max(height, 1u),
            .hzbMipCount = hzbMipCount_,
            .hzbElementCount = hzbElementCount_,
        };
        desc.visibleMeshletCapacity.fill(visibleMeshletCapacity);
        return desc;
    }

    Result refreshGPUSceneViewResources(std::string& log)
    {
        if (gpuSceneSubsystem_ == nullptr ||
            !gpuSceneView_.valid() ||
            frameSlotResources_.size() != frameSlotCount_) {
            log = "GPUDrivenPreviewPass has no GPUScene View allocation to refresh";
            return makeError(Error::InvalidArgument);
        }

        uint64_t allocationId = 0;
        for (uint32_t frameSlot = 0; frameSlot < frameSlotCount_; ++frameSlot) {
            GPUSceneViewGpuResourcesView resources;
            if (!gpuSceneSubsystem_->viewGpuResources(
                    gpuSceneView_,
                    frameSlot,
                    resources)) {
                log = "GPUDrivenPreviewPass failed to query GPUScene View resources";
                return makeError(Error::Failure);
            }
            if (allocationId == 0) {
                allocationId = resources.allocationId;
                hzbBuffers_[0] = resources.hzbHistory[0].buffer;
                hzbBuffers_[1] = resources.hzbHistory[1].buffer;
            } else if (allocationId != resources.allocationId ||
                       hzbBuffers_[0] != resources.hzbHistory[0].buffer ||
                       hzbBuffers_[1] != resources.hzbHistory[1].buffer) {
                log = "GPUDrivenPreviewPass received inconsistent GPUScene frame-slot resources";
                return makeError(Error::Failure);
            }

            GPUDrivenPreviewFrameSlotResources& slot = frameSlotResources_[frameSlot];
            slot.instanceVisibilityBuffer = resources.instanceVisibilityStates.buffer;
            slot.visibleInstanceIdsBuffer = resources.visibleInstanceIds.buffer;
            slot.visibleInstanceCounterBuffer = resources.visibleInstanceCounter.buffer;
            for (uint32_t phaseIndex = 0;
                 phaseIndex < kGPUSceneCullPhaseCount;
                 ++phaseIndex) {
                const GPUSceneCullPhaseGpuView& phase = resources.phases[phaseIndex];
                slot.visibleMeshletBuffers[phaseIndex] = phase.visibleMeshletIds.buffer;
                slot.indirectBuffers[phaseIndex] =
                    phase.buckets.front().indirectArguments.buffer;
                for (uint32_t bucketIndex = 0;
                     bucketIndex < kGPUSceneRasterDrawBucketCount;
                     ++bucketIndex) {
                    const GPUSceneBucketGpuView& bucket = phase.buckets[bucketIndex];
                    if (bucket.indirectArguments.buffer != slot.indirectBuffers[phaseIndex] ||
                        bucket.overflow.buffer != slot.indirectBuffers[phaseIndex] ||
                        bucket.visibleMeshletCapacity !=
                            resources.desc.visibleMeshletCapacity[bucketIndex] ||
                        bucket.visibleMeshletOffset !=
                            bucketIndex * resources.desc.visibleMeshletCapacity[bucketIndex]) {
                        log = "GPUDrivenPreviewPass received an incompatible GPUScene bucket layout";
                        return makeError(Error::Failure);
                    }
                }
            }
        }
        if (allocationId == 0 || hzbBuffers_[0] == nullptr || hzbBuffers_[1] == nullptr) {
            log = "GPUDrivenPreviewPass received an empty GPUScene View allocation";
            return makeError(Error::Failure);
        }
        gpuSceneViewAllocationId_ = allocationId;
        return {};
    }

    Result ensureGPUSceneViewResources(
        uint32_t width,
        uint32_t height,
        std::string& log)
    {
        if (gpuSceneSubsystem_ == nullptr || !gpuSceneView_.valid()) {
            log = "GPUDrivenPreviewPass requires a live GPUScene View";
            return makeError(Error::InvalidArgument);
        }
        Result result = gpuSceneSubsystem_->ensureViewGpuResources(
            gpuSceneView_,
            gpuSceneViewResourceDesc(width, height),
            log);
        return result ? refreshGPUSceneViewResources(log) : result;
    }

    Result syncGPUSceneRasterState(
        GPUSceneSubsystem& subsystem,
        std::string& log)
    {
        const GPUSceneDrawSet& drawSet = subsystem.drawSet();
        const GPUSceneGlobalBufferViews& views = subsystem.globalBufferViews();
        const GPUSceneRasterDrawLayout& layout = subsystem.rasterDrawLayout();
        if (drawSet.generation == 0 || drawSet.revision == 0 ||
            !views.validFor(drawSet.generation, drawSet.revision) ||
            !layout.validFor(drawSet.generation, drawSet.revision)) {
            log = "GPUDrivenPreviewPass requires current GPUScene raster buffers and layout";
            return makeError(Error::InvalidArgument);
        }
        if (layout.maxRangeCount == 0 || subsystem.instances().empty() ||
            subsystem.materials().empty()) {
            log = "GPUDrivenPreviewPass GPUScene raster layout has no drawable instances";
            return makeError(Error::Failure);
        }
        if (views.meshletDraws.structureStride != sizeof(VisibleClusterRecord) ||
            views.meshletDraws.size % sizeof(VisibleClusterRecord) != 0 ||
            views.meshletDraws.size / sizeof(VisibleClusterRecord) >
                std::numeric_limits<uint32_t>::max()) {
            log = "GPUDrivenPreviewPass received an invalid resident visible-record buffer";
            return makeError(Error::Failure);
        }
        const uint32_t residentRecordCapacity = static_cast<uint32_t>(
            views.meshletDraws.size / sizeof(VisibleClusterRecord));
        if (!visibilityRecordRangeFitsId(
                residentRecordCapacity,
                streamEnabled_ ? streamRuntime_.visibleClusterCapacity() : 0u)) {
            log = "GPUDrivenPreviewPass resident + stream records exceed the visibility ID range";
            return makeError(Error::Failure);
        }
        residentRecordCapacity_ = residentRecordCapacity;
        const auto encodedRangeValid = [](const GPUSceneRasterDrawRange& range) {
            return visibilityRecordCapacityFitsId(
                static_cast<uint64_t>(range.offset) + range.count);
        };
        if (!encodedRangeValid(layout.baseRange) ||
            !std::ranges::all_of(layout.lodRanges, encodedRangeValid)) {
            log = "GPUDrivenPreviewPass GPUScene raster layout exceeds the visibility ID range";
            return makeError(Error::Failure);
        }
        if (subsystem.materials().size() >
            std::numeric_limits<uint32_t>::max() /
                kGPUSceneMaterialTextureSlotCount) {
            log = "GPUDrivenPreviewPass GPUScene material remap is too large";
            return makeError(Error::Failure);
        }

        baseMeshletRange_ = GPUDrivenPreviewMeshletRange{
            .offset = layout.baseRange.offset,
            .count = layout.baseRange.count,
        };
        lodLevelRanges_.clear();
        lodLevelRanges_.reserve(layout.lodRanges.size());
        for (const GPUSceneRasterDrawRange& range : layout.lodRanges) {
            lodLevelRanges_.push_back(GPUDrivenPreviewMeshletRange{
                .offset = range.offset,
                .count = range.count,
            });
        }
        drawTaskCount_ = layout.maxRangeCount;
        instanceCount_ = static_cast<uint32_t>(subsystem.instances().size());
        if (streamOwnerMask_.size() != std::max(instanceCount_, 1u)) {
            streamOwnerMask_.assign(std::max(instanceCount_, 1u), 0u);
        }
        materialCount_ = static_cast<uint32_t>(subsystem.materials().size());
        materialTextureCount_ = materialCount_ * kGPUSceneMaterialTextureSlotCount;

        GPUSceneViewGpuResourcesView currentViewResources;
        if (!subsystem.viewGpuResources(gpuSceneView_, 0, currentViewResources)) {
            log = "GPUDrivenPreviewPass could not query its GPUScene View allocation";
            return makeError(Error::Failure);
        }
        constexpr uint32_t requiredVisibleMeshletCapacity = 1u;
        bool capacityGrowth =
            instanceCount_ > currentViewResources.desc.instanceCapacity;
        for (uint32_t bucketIndex = 0;
             bucketIndex < kGPUSceneRasterDrawBucketCount;
             ++bucketIndex) {
            capacityGrowth = capacityGrowth ||
                requiredVisibleMeshletCapacity >
                    currentViewResources.desc.visibleMeshletCapacity[bucketIndex];
        }
        if (capacityGrowth) {
            GPUSceneViewDesc grown = currentViewResources.desc;
            grown.instanceCapacity = std::max(
                grown.instanceCapacity,
                std::max(instanceCount_, 1u));
            for (uint32_t bucketIndex = 0;
                 bucketIndex < kGPUSceneRasterDrawBucketCount;
                 ++bucketIndex) {
                grown.visibleMeshletCapacity[bucketIndex] = std::max(
                    grown.visibleMeshletCapacity[bucketIndex],
                    requiredVisibleMeshletCapacity);
            }
            Result result = subsystem.ensureViewGpuResources(
                gpuSceneView_,
                grown,
                log);
            if (!result) {
                return result;
            }
            result = refreshGPUSceneViewResources(log);
            if (!result) {
                return result;
            }
        }
        return {};
    }

    Result ensureGPUSceneBindings(
        GPUSceneSubsystem& subsystem,
        RenderSubsystemHost& subsystemHost,
        std::string& log)
    {
        if (bindlessHeap_ == nullptr || materialTextureRemapBuffer_ == nullptr ||
            deferredColorBuffer_ == nullptr || cullingTargets_.depthView == nullptr) {
            log = "GPUDrivenPreviewPass cannot bind GPUScene before consumer resources exist";
            return makeError(Error::InvalidArgument);
        }
        const GPUSceneGlobalBufferViews& views = subsystem.globalBufferViews();
        if (!views.validFor(subsystem.drawSet().generation,
                            subsystem.drawSet().revision)) {
            log = "GPUDrivenPreviewPass cannot bind stale GPUScene global buffers";
            return makeError(Error::InvalidArgument);
        }
        const uint64_t expectedRemapBytes =
            static_cast<uint64_t>(materialTextureCount_) * sizeof(uint32_t);
        const bool remapLayoutChanged =
            materialTextureRemapBuffer_->desc().size != expectedRemapBytes;
        const uint64_t expectedOwnerMaskBytes = static_cast<uint64_t>(
            std::max(instanceCount_, 1u)) * sizeof(uint32_t);
        const bool ownerMaskLayoutChanged = streamEnabled_ &&
            (streamOwnerMaskBuffer_ == nullptr ||
                streamOwnerMaskBuffer_->desc().size != expectedOwnerMaskBytes);
        if (!gpuSceneBindings_.drawSetGeneration &&
            !remapLayoutChanged &&
            !streamEnabled_ &&
            !ownerMaskLayoutChanged) {
            return subsystem.createBindings(*bindlessHeap_, gpuSceneBindings_, log);
        }
        if (!remapLayoutChanged &&
            !ownerMaskLayoutChanged &&
            gpuSceneBindings_.validFor(views)) {
            return {};
        }

        GPUDrivenPreviewBindingBundle replacement;
        Result result = createBindingBundle(
            *deferredColorBuffer_,
            *cullingTargets_.depthView,
            true,
            replacement,
            log);
        if (!result) {
            return result;
        }
        auto retired = std::make_shared<GPUDrivenPreviewRetiredViewResources>();
        retired->bindlessHeap = std::move(bindlessHeap_);
        retired->materialTextureRemapBuffer =
            std::move(materialTextureRemapBuffer_);
        retired->streamDeferredBindingsBuffer =
            std::move(streamDeferredBindingsBuffer_);
        retired->streamOwnerMaskBuffer = std::move(streamOwnerMaskBuffer_);
        installBindingBundle(std::move(replacement));
        subsystemHost.retire(std::static_pointer_cast<void>(retired));
        return {};
    }

    GPUDrivenPreviewUserPush makePush(
        uint32_t passIndex = 0,
        uint32_t mipLevel = 0,
        bool projectWithCullingCamera = false) const
    {
        const GPUDrivenPreviewFrameSlotResources& slot = activeFrameResources();
        return GPUDrivenPreviewUserPush{
            .positionBuffer = gpuSceneBindings_[GPUSceneGlobalBufferKind::Vertices].index,
            .meshletBuffer = gpuSceneBindings_[GPUSceneGlobalBufferKind::Meshlets].index,
            .meshletDrawBuffer =
                gpuSceneBindings_[GPUSceneGlobalBufferKind::MeshletDraws].index,
            .meshletVertexBuffer =
                gpuSceneBindings_[GPUSceneGlobalBufferKind::MeshletVertices].index,
            .meshletTriangleBuffer =
                gpuSceneBindings_[GPUSceneGlobalBufferKind::MeshletTriangleWords].index,
            .paramsBuffer = slot.paramsHandle.index,
            .transformBuffer = gpuSceneBindings_[GPUSceneGlobalBufferKind::Geometries].index,
            .instanceBuffer = gpuSceneBindings_[GPUSceneGlobalBufferKind::Instances].index,
            .instanceVisibilityBuffer = slot.instanceVisibilityHandle.index,
            .visibleInstanceIdsBuffer = slot.visibleInstanceIdsHandle.index,
            .visibleInstanceCounterBuffer = slot.visibleInstanceCounterHandle.index,
            .visibleMeshletBuffer0 = slot.visibleMeshletHandles[0].index,
            .visibleMeshletBuffer1 = slot.visibleMeshletHandles[1].index,
            .indirectBuffer0 = slot.indirectHandles[0].index,
            .indirectBuffer1 = slot.indirectHandles[1].index,
            .hzbBuffer0 = hzbHandles_[0].index,
            .hzbBuffer1 = hzbHandles_[1].index,
            .deferredColorBuffer = deferredColorHandle_.index,
            .depthImage = freezeCullingCamera_
                ? cullingDepthImageHandle_.index
                : depthImageHandle_.index,
            .visibilityImage = visibilityImageHandle_.index,
            .passIndex = passIndex,
            .mipLevel = mipLevel,
            .projectWithCullingCamera = projectWithCullingCamera ? 1u : 0u,
            .materialBuffer = gpuSceneBindings_[GPUSceneGlobalBufferKind::Materials].index,
            .materialTextureRemapBuffer = materialTextureRemapHandle_.index,
            .environmentImage = environmentTextureHandle_.index,
            .environmentSHBuffer = environmentSHBufferHandle_.index,
            .streamDeferredBindingsBuffer =
                streamEnabled_ && streamDeferredBindingsHandle_.valid()
                ? streamDeferredBindingsHandle_.index
                : kGPUDrivenInvalidBindlessIndex,
            .residentRecordCapacity = residentRecordCapacity_,
            .streamOwnerMaskBuffer =
                streamEnabled_ && streamOwnerMaskHandle_.valid()
                ? streamOwnerMaskHandle_.index
                : kGPUDrivenInvalidBindlessIndex,
        };
    }

    static void transitionTexture(
        CommandBuffer& commandBuffer,
        Texture& texture,
        ResourceState before,
        ResourceState after)
    {
        const TextureBarrierDesc barrier{
            .texture = &texture,
            .before = before,
            .after = after,
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = &barrier,
            .textureCount = 1,
        });
    }

    static void barrierBuffer(
        CommandBuffer& commandBuffer,
        Buffer& buffer,
        ResourceState before,
        ResourceState after)
    {
        const BufferBarrierDesc barrier{
            .buffer = &buffer,
            .before = before,
            .after = after,
        };
        commandBuffer.barrier(BarrierDesc{
            .buffers = &barrier,
            .bufferCount = 1,
        });
    }

    Result initializeInternalBuffers(CommandBuffer& commandBuffer)
    {
        if (gpuSceneSubsystem_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        std::string log;
        Result result = gpuSceneSubsystem_->recordInitialize(
            commandBuffer,
            gpuSceneView_,
            activeFrameSlot_,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
            return result;
        }

        if (!frameBuffersInitialized_) {
            const BufferBarrierDesc barrier{
                .buffer = deferredColorBuffer_.get(),
                .before = ResourceState::Undefined,
                .after = ResourceState::General,
            };
            commandBuffer.barrier(BarrierDesc{
                .buffers = &barrier,
                .bufferCount = 1,
            });
            frameBuffersInitialized_ = true;
        }

        if (!cullingTargetsInitialized_) {
            const std::array<TextureBarrierDesc, 2> barriers{
                TextureBarrierDesc{
                    .texture = cullingTargets_.visibility.get(),
                    .before = ResourceState::Undefined,
                    .after = ResourceState::ColorAttachment,
                },
                TextureBarrierDesc{
                    .texture = cullingTargets_.depth.get(),
                    .before = ResourceState::Undefined,
                    .after = ResourceState::DepthStencilAttachment,
                },
            };
            commandBuffer.barrier(BarrierDesc{
                .textures = barriers.data(),
                .textureCount = static_cast<uint32_t>(barriers.size()),
            });
            cullingTargetsInitialized_ = true;
        }
        return {};
    }

    Result dispatchCulling(CommandBuffer& commandBuffer, uint32_t passIndex)
    {
        if (gpuSceneSubsystem_ == nullptr || passIndex >= kGPUSceneCullPhaseCount) {
            return makeError(Error::InvalidArgument);
        }
        GPUDrivenPreviewUserPush push = makePush(passIndex);
        const GPUSceneInstanceCullRecordDesc desc{
            .phase = passIndex == 0
                ? GPUSceneCullPhase::Early
                : GPUSceneCullPhase::Late,
            .bindlessHeap = bindlessHeap_.get(),
            .resetPipeline = resetPipeline_.get(),
            .instanceCullPipeline = instanceCullPipeline_.get(),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .instanceGroupCountX = divideRoundUp(instanceCount_, 64u),
        };
        std::string log;
        Result result = gpuSceneSubsystem_->recordInstanceCull(
            commandBuffer,
            gpuSceneView_,
            activeFrameSlot_,
            desc,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
        }
        return result;
    }

    void drawVisibility(
        CommandBuffer& commandBuffer,
        TextureView& visibility,
        TextureView& depth,
        uint32_t passIndex,
        LoadOp loadOp,
        bool projectWithCullingCamera = false)
    {
        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = frameWidth_,
            .height = frameHeight_,
        };
        const RenderingAttachmentDesc visibilityAttachment{
            .view = &visibility,
            .state = ResourceState::ColorAttachment,
            .loadOp = loadOp,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 0.0f},
        };
        const RenderingAttachmentDesc depthAttachment{
            .view = &depth,
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = loadOp,
            .storeOp = StoreOp::Store,
            .clearDepth = depthClearValue(kDefaultReversedZ),
        };
        commandBuffer.beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &visibilityAttachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = &depthAttachment,
        });
        commandBuffer.setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(frameWidth_),
            .height = static_cast<float>(frameHeight_),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        commandBuffer.setScissor(renderArea);
        commandBuffer.bindBindlessHeap(*bindlessHeap_);
        for (uint32_t bucketIndex = 0;
             bucketIndex < kGPUDrivenPreviewDrawBucketCount;
             ++bucketIndex) {
            commandBuffer.bindGraphicsPipeline(*visibilityPipelines_[bucketIndex]);
            const GPUDrivenPreviewUserPush push =
                makePush(passIndex, bucketIndex, projectWithCullingCamera);
            commandBuffer.pushBindlessData(&push, sizeof(push));
            commandBuffer.drawMeshTasks(
                divideRoundUp(
                    activeMeshletCount_,
                    kGPUDrivenPreviewAmplificationGroupSize));
        }
        commandBuffer.endRendering();
    }

    Result buildHzb(CommandBuffer& commandBuffer)
    {
        if (gpuSceneSubsystem_ == nullptr || hzbMipCount_ == 0) {
            return makeError(Error::InvalidArgument);
        }
        std::vector<GPUDrivenPreviewUserPush> pushes;
        pushes.reserve(hzbMipCount_);
        for (uint32_t mipLevel = 0; mipLevel < hzbMipCount_; ++mipLevel) {
            pushes.push_back(makePush(0, mipLevel));
        }
        std::vector<GPUSceneComputeDispatchDesc> dispatches;
        dispatches.reserve(hzbMipCount_);
        uint32_t mipWidth = frameWidth_;
        uint32_t mipHeight = frameHeight_;
        for (uint32_t mipLevel = 0; mipLevel < hzbMipCount_; ++mipLevel) {
            dispatches.push_back(GPUSceneComputeDispatchDesc{
                .pushData = &pushes[mipLevel],
                .pushDataSize = sizeof(pushes[mipLevel]),
                .groupCountX = divideRoundUp(mipWidth, 8u),
                .groupCountY = divideRoundUp(mipHeight, 8u),
                .groupCountZ = 1,
            });
            mipWidth = std::max(1u, (mipWidth + 1u) / 2u);
            mipHeight = std::max(1u, (mipHeight + 1u) / 2u);
        }
        const GPUSceneHzbRecordDesc desc{
            .bindlessHeap = bindlessHeap_.get(),
            .pipeline = hzbPipeline_.get(),
            .dispatches = dispatches,
        };
        std::string log;
        Result result = gpuSceneSubsystem_->recordBuildHzb(
            commandBuffer,
            gpuSceneView_,
            activeFrameSlot_,
            desc,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
        }
        return result;
    }

    void dispatchDeferred(CommandBuffer& commandBuffer)
    {
        const GPUDrivenPreviewUserPush push = makePush();
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.bindComputePipeline(*deferredPipeline_);
        commandBuffer.dispatch(divideRoundUp(frameWidth_, 8u), divideRoundUp(frameHeight_, 8u), 1);
    }

    void drawComposite(CommandBuffer& commandBuffer, TextureHandle color)
    {
        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = frameWidth_,
            .height = frameHeight_,
        };
        const RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.015f, 0.018f, 0.024f, 1.0f},
        };
        commandBuffer.beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        commandBuffer.setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(frameWidth_),
            .height = static_cast<float>(frameHeight_),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        commandBuffer.setScissor(renderArea);
        commandBuffer.bindGraphicsPipeline(*compositePipeline_);
        const GPUDrivenPreviewUserPush push = makePush();
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.draw(3);
        commandBuffer.endRendering();
    }

    Result prepareGPUSceneView(
        GPUSceneSubsystem& subsystem,
        bool cameraCut)
    {
        const GPUSceneViewPrepareInfo prepareInfo{
            .width = frameWidth_,
            .height = frameHeight_,
            .cameraCut = cameraCut,
            .freezeCullingCamera = boolProperty(
                &properties(),
                "freezeCullingCamera",
                false),
        };
        if (!subsystem.prepareView(gpuSceneView_, prepareInfo)) {
            return makeError(Error::InvalidArgument);
        }
        const GPUSceneVisibleDrawSet* visibleDrawSet =
            subsystem.visibleDrawSet(gpuSceneView_, activeFrameSlot_);
        if (visibleDrawSet == nullptr) {
            return makeError(Error::Failure);
        }
        if (!visibleDrawSet->stats.hzbValid) {
            hzbValid_ = false;
        } else if (!hzbValid_) {
            if (!subsystem.markViewHzbValid(gpuSceneView_, activeFrameSlot_, false) ||
                !subsystem.prepareView(gpuSceneView_, prepareInfo)) {
                return makeError(Error::Failure);
            }
            visibleDrawSet = subsystem.visibleDrawSet(gpuSceneView_, activeFrameSlot_);
            if (visibleDrawSet == nullptr) {
                return makeError(Error::Failure);
            }
        }
        const uint32_t drawSetGeneration = subsystem.drawSet().generation;
        const uint64_t drawSetRevision = subsystem.drawSet().revision;
        if (drawSetRevision == 0 ||
            activeFrameResources().instanceVisibilityBuffer == nullptr ||
            activeFrameResources().indirectBuffers[0] == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (drawSetGeneration != gpuSceneDrawSetGeneration_) {
            gpuSceneDrawSetGeneration_ = drawSetGeneration;
            invalidateHzbHistory();
        }
        gpuSceneDrawSetRevision_ = drawSetRevision;
        std::string log;
        Result result = subsystem.publishViewGpuResources(
            gpuSceneView_,
            activeFrameSlot_,
            (streamEnabled_ ? streamRuntime_.frameIndex() : frameIndex_) & 1u,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
        }
        return result;
    }

    Result syncRuntimeGeometry(const scene::Scene* runtimeScene)
    {
        runtimeScene = runtimeSceneForPath(runtimeScene, scenePathFromProperties(properties()));
        if (runtimeScene == nullptr) {
            return {};
        }
        const bool transformsChanged = runtimeScene->transformRevision() != sceneRevision_;
        const bool visibilityChanged =
            runtimeScene->visibilityRevision() != sceneVisibilityRevision_;
        if (!transformsChanged && !visibilityChanged) {
            return {};
        }

        if (transformsChanged) {
            drawBounds_ = runtimeScene->bounds();
            sceneRevision_ = runtimeScene->transformRevision();
        }

        if (visibilityChanged) {
            sceneVisibilityRevision_ = runtimeScene->visibilityRevision();
        }
        invalidateHzbHistory();
        return {};
    }

    Result syncStreamRuntimeScene(
        const scene::Scene* runtimeScene,
        GPUSceneSubsystem& subsystem,
        std::string& log)
    {
        if (!streamEnabled_) {
            return {};
        }
        runtimeScene = runtimeSceneForPath(
            runtimeScene,
            scenePathFromProperties(properties()));
        if (runtimeScene == nullptr || !streamRuntime_.ready()) {
            log = "GPUDrivenPreviewPass stream integration requires its runtime scene";
            return makeError(Error::InvalidArgument);
        }
        std::vector<uint32_t> runtimeRenderNodeIndices(
            streamRuntime_.asset().instances().size(),
            std::numeric_limits<uint32_t>::max());
        for (size_t streamInstanceIndex = 0;
             streamInstanceIndex < streamRuntime_.asset().instances().size();
             ++streamInstanceIndex) {
            const uint32_t localRenderNodeIndex =
                streamRuntime_.asset().instances()[streamInstanceIndex].renderNodeIndex;
            if (compiledStreamSourceId_.empty()) {
                runtimeRenderNodeIndices[streamInstanceIndex] = localRenderNodeIndex;
                continue;
            }
            const int32_t runtimeRenderNodeIndex = runtimeScene->renderNodeIndexForSource(
                compiledStreamSourceId_,
                localRenderNodeIndex <=
                        static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
                    ? static_cast<int32_t>(localRenderNodeIndex)
                    : scene::kInvalidSceneIndex);
            if (runtimeRenderNodeIndex == scene::kInvalidSceneIndex) {
                log = "GPUDrivenPreviewPass stream source '" +
                    compiledStreamSourceId_ + "' has no render node " +
                    std::to_string(localRenderNodeIndex);
                return makeError(Error::InvalidArgument);
            }
            runtimeRenderNodeIndices[streamInstanceIndex] =
                static_cast<uint32_t>(runtimeRenderNodeIndex);
        }

        Result result = streamRuntime_.syncRuntimeScene(
            *runtimeScene,
            runtimeRenderNodeIndices,
            log);
        if (!result) {
            return result;
        }

        std::vector<uint32_t> mapping(
            streamRuntime_.asset().instances().size(),
            std::numeric_limits<uint32_t>::max());
        streamOwnerMask_.assign(std::max(instanceCount_, 1u), 0u);
        uint32_t mappedCount = 0;
        for (size_t streamInstanceIndex = 0;
             streamInstanceIndex < streamRuntime_.asset().instances().size();
             ++streamInstanceIndex) {
            const GPUSceneInstanceId instance = subsystem.instanceForRenderNode(
                runtimeRenderNodeIndices[streamInstanceIndex]);
            if (!instance.valid() || instance.index >= streamOwnerMask_.size()) {
                continue;
            }
            mapping[streamInstanceIndex] = instance.index;
            streamOwnerMask_[instance.index] = 1u;
            ++mappedCount;
        }
        result = streamRuntime_.syncGPUSceneInstanceMapping(mapping);
        if (!result) {
            log = "GPUDrivenPreviewPass failed to upload the stream GPUScene mapping";
            return result;
        }
        streamMappedInstanceCount_ = mappedCount;
        return {};
    }

    uint32_t streamCullingFlags() const
    {
        uint32_t flags = 0;
        if (boolProperty(&properties(), "instanceFrustumCull", true)) {
            flags |= 1u << 0u;
        }
        if (boolProperty(&properties(), "instanceHzbCull", true)) {
            flags |= 1u << 1u;
        }
        if (boolProperty(&properties(), "meshletFrustumCull", true)) {
            flags |= 1u << 2u;
        }
        if (boolProperty(&properties(), "meshletNormalConeCull", true)) {
            flags |= 1u << 3u;
        }
        return flags;
    }

    Result bindStreamViewResources(
        GPUSceneSubsystem& subsystem,
        TextureHandle visibility,
        TextureHandle depth)
    {
        if (!streamEnabled_ || streamRuntime_.bindlessHeap() == nullptr ||
            !visibility.valid() || !depth.valid()) {
            return makeError(Error::InvalidArgument);
        }
        GPUSceneViewGpuResourcesView resources;
        if (!subsystem.viewGpuResources(
                gpuSceneView_,
                activeFrameSlot_,
                resources) ||
            resources.instanceVisibilityStates.view == nullptr ||
            resources.visibleInstanceIds.view == nullptr ||
            resources.visibleInstanceCounter.view == nullptr ||
            resources.hzbHistory[0].view == nullptr ||
            resources.hzbHistory[1].view == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        BindlessHeap& heap = *streamRuntime_.bindlessHeap();
        Result result = heap.writeSampledImage(
            streamVisibilityImageHandle_,
            *visibility.view(),
            ResourceState::ShaderRead);
        if (result) {
            result = heap.writeSampledImage(
                streamDepthImageHandle_,
                *depth.view(),
                ResourceState::ShaderRead);
        }
        if (result) {
            result = heap.writeBufferView(
                streamInstanceVisibilityHandle_,
                *resources.instanceVisibilityStates.view);
        }
        if (result) {
            result = heap.writeBufferView(
                streamVisibleInstanceIdsHandle_,
                *resources.visibleInstanceIds.view);
        }
        if (result) {
            result = heap.writeBufferView(
                streamVisibleInstanceCounterHandle_,
                *resources.visibleInstanceCounter.view);
        }
        for (uint32_t historyIndex = 0;
             historyIndex < streamHzbHandles_.size() && result;
             ++historyIndex) {
            result = heap.writeBufferView(
                streamHzbHandles_[historyIndex],
                *resources.hzbHistory[historyIndex].view);
        }
        if (!result) {
            return result;
        }

        const uint32_t streamRecordCapacity =
            streamRuntime_.visibleClusterCapacity();
        if (!visibilityRecordRangeFitsId(
                residentRecordCapacity_,
                streamRecordCapacity)) {
            return makeError(Error::InvalidArgument);
        }
        result = streamRuntime_.updateRasterBindings(
            MeshletStreamGpuRasterBindings{
                .instanceVisibilityBuffer =
                    streamInstanceVisibilityHandle_.index,
                .hzbBuffer0 = streamHzbHandles_[0].index,
                .hzbBuffer1 = streamHzbHandles_[1].index,
                .depthImage = streamDepthImageHandle_.index,
                .visibilityImage = streamVisibilityImageHandle_.index,
                .visibleInstanceIdsBuffer =
                    streamVisibleInstanceIdsHandle_.index,
                .visibleRecordBase = residentRecordCapacity_,
                .visibleRecordCapacity = streamRecordCapacity,
                .hzbMipCount = hzbMipCount_,
                .hzbValid = hzbValid_ ? 1u : 0u,
                .cullingFlags = streamCullingFlags(),
                .width = frameWidth_,
                .height = frameHeight_,
                .visibleInstanceCounterBuffer =
                    streamVisibleInstanceCounterHandle_.index,
            });
        if (!result || streamDeferredBindingsBuffer_ == nullptr ||
            streamOwnerMaskBuffer_ == nullptr) {
            return result ? makeError(Error::InvalidArgument) : result;
        }
        const GPUDrivenStreamDeferredBindings deferredBindings{
            .pageBuffer = streamDeferredResourceHandles_[
                GPUDrivenStreamDeferredPage].index,
            .activeGroupBuffer = streamDeferredResourceHandles_[
                GPUDrivenStreamDeferredActiveGroup].index,
            .pageTableBuffer = streamDeferredResourceHandles_[
                GPUDrivenStreamDeferredPageTable].index,
            .activeHeaderBuffer = streamDeferredResourceHandles_[
                GPUDrivenStreamDeferredActiveHeader].index,
            .paramsBuffer = streamDeferredResourceHandles_[
                GPUDrivenStreamDeferredParams].index,
            .visibleClusterBuffer = streamDeferredResourceHandles_[
                GPUDrivenStreamDeferredVisibleCluster].index,
            .visibleRecordBase = residentRecordCapacity_,
            .visibleRecordCapacity = streamRecordCapacity,
        };
        result = updateHostStorageBuffer(
            *streamDeferredBindingsBuffer_,
            &deferredBindings,
            sizeof(deferredBindings));
        if (result) {
            result = updateHostStorageBuffer(
                *streamOwnerMaskBuffer_,
                streamOwnerMask_.data(),
                static_cast<uint64_t>(streamOwnerMask_.size()) * sizeof(uint32_t));
        }
        return result;
    }

    MeshletStreamFrameDesc streamFrameDesc(
        const RenderGraphExecutionContext& context) const
    {
        const scene::Bounds& bounds = streamRuntime_.bounds();
        const float3 center = bounds.center();
        const float radius = std::max(bounds.radius(), 1.0f);
        const RenderGraphProperties* camera =
            cameraPropertiesFrom(context.properties());
        const float3 defaultEye(
            center.x,
            center.y + radius * 0.35f,
            center.z + radius * 2.5f);
        const bool gpuLod = boolProperty(
            &context.properties(),
            "enableGpuLodSelection",
            true);
        uint32_t debugMode = kMeshletStreamDebugShaded;
        switch (previewModeFromProperties(context.properties())) {
        case kGPUDrivenPreviewModeMeshlet:
            debugMode = kMeshletStreamDebugMeshlet;
            break;
        case kGPUDrivenPreviewModePrimitive:
            debugMode = kMeshletStreamDebugPrimitive;
            break;
        case kGPUDrivenPreviewModeLod:
            debugMode = kMeshletStreamDebugLod;
            break;
        default:
            break;
        }
        return MeshletStreamFrameDesc{
            .width = frameWidth_,
            .height = frameHeight_,
            .selectedLodLevel = gpuLod
                ? kMeshletStreamNoDebugLodOverride
                : lodLevelFromProperties(context.properties()),
            .enableGpuLodSelection = gpuLod,
            .debugColorMode = debugMode,
            .camera = MeshletStreamCameraDesc{
                .eye = cameraVec3(camera, "eye", defaultEye),
                .center = cameraVec3(camera, "center", center),
                .up = cameraVec3(camera, "up", float3(0.0f, 1.0f, 0.0f)),
                .fovDegrees = cameraFloat(camera, "fovDegrees", 60.0f),
                .znear = cameraFloat(camera, "znear", 0.1f),
                .zfar = cameraFloat(
                    camera,
                    "zfar",
                    std::max(radius * 8.0f, 100.0f)),
            },
        };
    }

    Result dispatchStreamCulling(
        CommandBuffer& commandBuffer,
        GPUSceneCullPhase phase)
    {
        if (!streamEnabled_ || gpuSceneSubsystem_ == nullptr ||
            streamCullResetPipeline_ == nullptr ||
            streamInstanceCullPipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        MeshletStreamUserPush push = streamRuntime_.userPush();
        push.traversalPhase = phase == GPUSceneCullPhase::Early ? 0u : 1u;
        const GPUSceneInstanceCullRecordDesc desc{
            .phase = phase,
            .bindlessHeap = streamRuntime_.bindlessHeap(),
            .resetPipeline = streamCullResetPipeline_.get(),
            .instanceCullPipeline = streamInstanceCullPipeline_.get(),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .instanceGroupCountX = std::max(
                divideRoundUp(
                    static_cast<uint32_t>(
                        streamRuntime_.asset().instances().size()),
                    64u),
                1u),
        };
        std::string log;
        Result result = gpuSceneSubsystem_->recordInstanceCull(
            commandBuffer,
            gpuSceneView_,
            activeFrameSlot_,
            desc,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
        }
        return result;
    }

    Result drawStreamVisibility(
        CommandBuffer& commandBuffer,
        TextureView& visibility,
        TextureView& depth,
        GPUSceneCullPhase phase,
        LoadOp loadOp)
    {
        if (!streamEnabled_ || streamVisibilityPipeline_ == nullptr ||
            streamRuntime_.bindlessHeap() == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        Result result = streamRuntime_.cmdPrepareVisibility(commandBuffer);
        if (!result) {
            return result;
        }
        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = frameWidth_,
            .height = frameHeight_,
        };
        const RenderingAttachmentDesc visibilityAttachment{
            .view = &visibility,
            .state = ResourceState::ColorAttachment,
            .loadOp = loadOp,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 0.0f},
        };
        const RenderingAttachmentDesc depthAttachment{
            .view = &depth,
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = loadOp,
            .storeOp = StoreOp::Store,
            .clearDepth = depthClearValue(kDefaultReversedZ),
        };
        commandBuffer.beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &visibilityAttachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = &depthAttachment,
        });
        commandBuffer.setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(frameWidth_),
            .height = static_cast<float>(frameHeight_),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        commandBuffer.setScissor(renderArea);
        commandBuffer.bindBindlessHeap(*streamRuntime_.bindlessHeap());
        commandBuffer.bindGraphicsPipeline(*streamVisibilityPipeline_);
        MeshletStreamUserPush push = streamRuntime_.userPush();
        push.traversalPhase = phase == GPUSceneCullPhase::Early ? 0u : 1u;
        commandBuffer.pushBindlessData(&push, sizeof(push));
        streamRuntime_.cmdDrawMeshTasks(commandBuffer);
        commandBuffer.endRendering();
        return {};
    }

    Result prepareShadingResources(
        Device& device,
        const scene::Scene& loadedScene,
        const RenderGraphProperties& properties,
        std::vector<GPUDrivenPreviewGpuMaterial>& outMaterials,
        std::string& log)
    {
        materialTextures_.clear();
        materialTextureHandles_.clear();
        logicalTextureToMaterialTexture_.clear();
        environmentTextureHandle_ = {};
        environmentSHBufferHandle_ = {};
        openPBRLut2D_ = {};
        openPBRLut3D_ = {};
        openPBRLut2DHandles_ = {};
        openPBRLut3DHandles_ = {};
        environmentResourceRevision_ = 0;

        const uint8_t whitePixel[4] = {255u, 255u, 255u, 255u};
        GPUDrivenPreviewTextureResource fallbackTexture;
        Result result = createGPUDrivenTexture(
            device,
            whitePixel,
            sizeof(whitePixel),
            1,
            1,
            Format::Rgba8Unorm,
            "GPUDrivenPreviewPass material fallback",
            fallbackTexture,
            log);
        if (!result) {
            return result;
        }
        materialTextures_.push_back(std::move(fallbackTexture));

        std::vector<uint32_t> textureIndexMap(
            loadedScene.textures().size(),
            std::numeric_limits<uint32_t>::max());

        struct MaterialDecodeTaskResult {
            GPUDrivenPreviewDecodedImage image;
            std::string warning;
            bool decoded = false;
        };
        const size_t textureCount = loadedScene.textures().size();
        const size_t hardwareThreads =
            std::max<size_t>(std::thread::hardware_concurrency(), 1u);
        const size_t decodeWorkerLimit = std::min<size_t>(
            8u,
            std::max<size_t>(hardwareThreads / 2u, 1u));
        double materialDecodeWallMilliseconds = 0.0;
        double materialTextureCreateMilliseconds = 0.0;
        size_t attemptedTextureCount = 0;
        for (size_t batchBegin = 0;
             batchBegin < textureCount && materialTextures_.size() < kGPUDrivenMaxMaterialTextures;
             batchBegin += decodeWorkerLimit) {
            const size_t batchCount = std::min(decodeWorkerLimit, textureCount - batchBegin);
            attemptedTextureCount += batchCount;
            std::vector<MaterialDecodeTaskResult> decodedBatch(batchCount);
            std::atomic_size_t nextDecode{0};
            const auto decodeBatchBegin = GPUDrivenCompileClock::now();
            const auto processDecodeTasks = [&]() {
                for (;;) {
                    const size_t localIndex = nextDecode.fetch_add(1, std::memory_order_relaxed);
                    if (localIndex >= batchCount) {
                        return;
                    }
                    MaterialDecodeTaskResult& decoded = decodedBatch[localIndex];
                    decoded.decoded = decodeGPUDrivenMaterialTexture(
                        loadedScene,
                        static_cast<uint32_t>(batchBegin + localIndex),
                        decoded.image,
                        decoded.warning);
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(batchCount > 0 ? batchCount - 1u : 0u);
            for (size_t workerIndex = 1; workerIndex < batchCount; ++workerIndex) {
                workers.emplace_back(processDecodeTasks);
            }
            processDecodeTasks();
            for (std::thread& worker : workers) {
                worker.join();
            }
            materialDecodeWallMilliseconds += std::chrono::duration<double, std::milli>(
                GPUDrivenCompileClock::now() - decodeBatchBegin).count();

            for (size_t localIndex = 0; localIndex < batchCount; ++localIndex) {
                if (materialTextures_.size() >= kGPUDrivenMaxMaterialTextures) {
                    break;
                }
                MaterialDecodeTaskResult& decoded = decodedBatch[localIndex];
                log += decoded.warning;
                if (!decoded.decoded) {
                    continue;
                }
                GPUDrivenPreviewTextureResource texture;
                const auto textureCreateBegin = GPUDrivenCompileClock::now();
                result = createGPUDrivenTexture(
                    device,
                    decoded.image.pixels.data(),
                    static_cast<uint64_t>(decoded.image.pixels.size()),
                    decoded.image.width,
                    decoded.image.height,
                    Format::Rgba8Unorm,
                    "GPUDrivenPreviewPass material texture",
                    texture,
                    log);
                if (!result) {
                    return result;
                }
                materialTextureCreateMilliseconds += std::chrono::duration<double, std::milli>(
                    GPUDrivenCompileClock::now() - textureCreateBegin).count();
                const size_t textureIndex = batchBegin + localIndex;
                textureIndexMap[textureIndex] = static_cast<uint32_t>(materialTextures_.size());
                materialTextures_.push_back(std::move(texture));
            }
        }
        spdlog::info(
            "[GPUDrivenPreviewPass] Processed {} material texture decodes with {} workers in {:.2f} ms and created textures in {:.2f} ms",
            attemptedTextureCount,
            decodeWorkerLimit,
            materialDecodeWallMilliseconds,
            materialTextureCreateMilliseconds);
        logicalTextureToMaterialTexture_ = textureIndexMap;

        outMaterials.clear();
        outMaterials.reserve(std::max<size_t>(loadedScene.materials().size(), 1u));
        if (loadedScene.materials().empty()) {
            outMaterials.push_back(GPUDrivenPreviewGpuMaterial{});
        } else {
            for (const scene::RenderMaterial& material : loadedScene.materials()) {
                outMaterials.push_back(gpuDrivenMaterial(material, textureIndexMap));
            }
        }

        auto shadingStageBegin = GPUDrivenCompileClock::now();
        result = prepareGPUDrivenOpenPBRLuts(device, openPBRLut2D_, openPBRLut3D_, log);
        logGPUDrivenCompileStage("OpenPBR LUT textures", shadingStageBegin);
        return result;
    }

    Result uploadShadingTextures(CommandBuffer& commandBuffer)
    {
        for (GPUDrivenPreviewTextureResource& texture : materialTextures_) {
            Result result = uploadGPUDrivenTexture(commandBuffer, texture);
            if (!result) {
                return result;
            }
        }
        Result result;
        for (GPUDrivenPreviewTextureResource& texture : openPBRLut2D_) {
            result = uploadGPUDrivenTexture(commandBuffer, texture);
            if (!result) {
                return result;
            }
        }
        for (GPUDrivenPreviewTextureResource &texture : openPBRLut3D_) {
            result = uploadGPUDrivenTexture(commandBuffer, texture);
            if (!result) {
                return result;
            }
        }
        return {};
    }

    static Result uploadStorageBuffer(
        Device& device,
        const void* data,
        uint64_t byteSize,
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
                .structureStride = 0,
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

    static Result updateHostStorageBuffer(
        Buffer& buffer,
        const void* data,
        uint64_t byteSize)
    {
        if (data == nullptr || byteSize == 0 || byteSize > buffer.desc().size) {
            return makeError(Error::InvalidArgument);
        }
        void* mapped = buffer.map();
        if (mapped == nullptr) {
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, data, static_cast<size_t>(byteSize));
        buffer.flush(0, byteSize);
        buffer.unmap();
        return {};
    }

    static Result allocateAndWriteBuffer(
        BindlessHeap& heap,
        Buffer& buffer,
        BindlessHandle& outHandle,
        std::string& log,
        std::string_view label)
    {
        Result result = heap.allocateBuffer(outHandle);
        if (!result || !outHandle.valid()) {
            log += resultMessage(std::string("allocateBuffer(GPUDrivenPreviewPass ") +
                                     std::string(label) + ")",
                                 result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = heap.writeStorageBuffer(outHandle, buffer);
        if (!result) {
            log += resultMessage(std::string("writeStorageBuffer(GPUDrivenPreviewPass ") +
                                     std::string(label) + ")",
                                 result);
            log += '\n';
        }
        return result;
    }

    Result createBindingBundle(
        Buffer& deferredColorBuffer,
        TextureView& cullingDepthView,
        bool includeGPUSceneBindings,
        GPUDrivenPreviewBindingBundle& outBundle,
        std::string& log)
    {
        if (device_ == nullptr || gpuSceneSubsystem_ == nullptr ||
            frameSlotResources_.size() != frameSlotCount_ || hzbBuffers_[0] == nullptr ||
            hzbBuffers_[1] == nullptr) {
            log = "GPUDrivenPreviewPass cannot build bindings before GPU resources "
                  "are ready";
            return makeError(Error::InvalidArgument);
        }

        GPUDrivenPreviewBindingBundle bundle;
        Result result = device_->createBindlessHeap(
            BindlessHeapDesc{
                .maxSampledImages = 4u + static_cast<uint32_t>(materialTextures_.size()) +
                                    kGPUDrivenOpenPBRLut2DCount +
                                    kGPUDrivenOpenPBRLut3DCount,
                .maxBuffers = 5u + frameSlotCount_ * 8u +
                    static_cast<uint32_t>(kGPUSceneGlobalBufferKindCount) +
                    (streamEnabled_ ? 8u : 0u),
            },
            bundle.heap);
        if (!result || bundle.heap == nullptr) {
            log += resultMessage("createBindlessHeap(GPUDrivenPreviewPass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        auto bindBuffer = [&](Buffer& buffer, BindlessHandle& handle,
                              std::string_view label) -> Result {
            return allocateAndWriteBuffer(*bundle.heap, buffer, handle, log, label);
        };
        auto allocateImage = [&](BindlessHandle& handle, std::string_view label) -> Result {
            Result allocateResult = bundle.heap->allocateSampledImage(handle);
            if (!allocateResult || !handle.valid()) {
                log += resultMessage(std::string("allocateSampledImage(GPUDrivenPreviewPass ") +
                                         std::string(label) + ")",
                                     allocateResult);
                log += '\n';
                return allocateResult ? makeError(Error::Failure) : allocateResult;
            }
            return {};
        };
        auto writeImage = [&](BindlessHandle handle, TextureView& view,
                              std::string_view label) -> Result {
            Result writeResult =
                bundle.heap->writeSampledImage(handle, view, ResourceState::ShaderRead);
            if (!writeResult) {
                log += resultMessage(std::string("writeSampledImage(GPUDrivenPreviewPass ") +
                                         std::string(label) + ")",
                                     writeResult);
                log += '\n';
            }
            return writeResult;
        };

        bundle.frameSlots.resize(frameSlotResources_.size());
        for (size_t frameSlot = 0; frameSlot < frameSlotResources_.size(); ++frameSlot) {
            const GPUDrivenPreviewFrameSlotResources& resources = frameSlotResources_[frameSlot];
            GPUDrivenPreviewFrameSlotBindings& bindings = bundle.frameSlots[frameSlot];
            if (resources.paramsBuffer == nullptr ||
                resources.instanceVisibilityBuffer == nullptr ||
                resources.visibleInstanceIdsBuffer == nullptr ||
                resources.visibleInstanceCounterBuffer == nullptr) {
                log = "GPUDrivenPreviewPass frame-slot binding resource is null";
                return makeError(Error::InvalidArgument);
            }
            result =
                bindBuffer(*resources.paramsBuffer, bindings.paramsHandle, "frame-slot params");
            if (!result) {
                return result;
            }
            result =
                bindBuffer(*resources.instanceVisibilityBuffer, bindings.instanceVisibilityHandle,
                           "frame-slot instance visibility");
            if (!result) {
                return result;
            }
            result =
                bindBuffer(*resources.visibleInstanceIdsBuffer, bindings.visibleInstanceIdsHandle,
                           "frame-slot visible instance IDs");
            if (!result) {
                return result;
            }
            result = bindBuffer(*resources.visibleInstanceCounterBuffer,
                                bindings.visibleInstanceCounterHandle,
                                "frame-slot visible instance counter");
            if (!result) {
                return result;
            }
            for (uint32_t phaseIndex = 0; phaseIndex < kGPUSceneCullPhaseCount; ++phaseIndex) {
                if (resources.visibleMeshletBuffers[phaseIndex] == nullptr ||
                    resources.indirectBuffers[phaseIndex] == nullptr) {
                    log = "GPUDrivenPreviewPass cull-phase binding resource is null";
                    return makeError(Error::InvalidArgument);
                }
                result = bindBuffer(*resources.visibleMeshletBuffers[phaseIndex],
                                    bindings.visibleMeshletHandles[phaseIndex],
                                    "frame-slot visible meshlets");
                if (!result) {
                    return result;
                }
                result = bindBuffer(*resources.indirectBuffers[phaseIndex],
                                    bindings.indirectHandles[phaseIndex],
                                    "frame-slot indirect arguments");
                if (!result) {
                    return result;
                }
            }
        }

        for (uint32_t bufferIndex = 0; bufferIndex < hzbBuffers_.size(); ++bufferIndex) {
            result = bindBuffer(*hzbBuffers_[bufferIndex], bundle.hzbHandles[bufferIndex], "HZB");
            if (!result) {
                return result;
            }
        }
        result = bindBuffer(deferredColorBuffer, bundle.deferredColorHandle, "deferred color");
        if (!result) {
            return result;
        }
        if (streamEnabled_) {
            const MeshletStreamDeferredGpuResourcesView streamResources =
                streamRuntime_.deferredGpuResources();
            if (!streamResources.valid() || streamOwnerMask_.empty()) {
                log = "GPUDrivenPreviewPass stream deferred resources are incomplete";
                return makeError(Error::InvalidArgument);
            }
            const std::array<Buffer*, GPUDrivenStreamDeferredResourceCount>
                streamBuffers{
                    streamResources.pageBuffer,
                    streamResources.activeGroupBuffer,
                    streamResources.pageTableBuffer,
                    streamResources.activeHeaderBuffer,
                    streamResources.paramsBuffer,
                    streamResources.visibleClusterBuffer,
                };
            const std::array<const char*, GPUDrivenStreamDeferredResourceCount>
                streamLabels{
                    "stream pages",
                    "stream active groups",
                    "stream page table",
                    "stream active header",
                    "stream params",
                    "stream visible records",
                };
            for (uint32_t resourceIndex = 0;
                 resourceIndex < streamBuffers.size();
                 ++resourceIndex) {
                result = bindBuffer(
                    *streamBuffers[resourceIndex],
                    bundle.streamDeferredResourceHandles[resourceIndex],
                    streamLabels[resourceIndex]);
                if (!result) {
                    return result;
                }
            }

            const GPUDrivenStreamDeferredBindings streamBindings{
                .pageBuffer = bundle.streamDeferredResourceHandles[
                    GPUDrivenStreamDeferredPage].index,
                .activeGroupBuffer = bundle.streamDeferredResourceHandles[
                    GPUDrivenStreamDeferredActiveGroup].index,
                .pageTableBuffer = bundle.streamDeferredResourceHandles[
                    GPUDrivenStreamDeferredPageTable].index,
                .activeHeaderBuffer = bundle.streamDeferredResourceHandles[
                    GPUDrivenStreamDeferredActiveHeader].index,
                .paramsBuffer = bundle.streamDeferredResourceHandles[
                    GPUDrivenStreamDeferredParams].index,
                .visibleClusterBuffer = bundle.streamDeferredResourceHandles[
                    GPUDrivenStreamDeferredVisibleCluster].index,
                .visibleRecordBase = residentRecordCapacity_,
                .visibleRecordCapacity = streamResources.visibleRecordCapacity,
            };
            result = uploadStorageBuffer(
                *device_,
                &streamBindings,
                sizeof(streamBindings),
                bundle.streamDeferredBindingsBuffer,
                log,
                "GPUDrivenPreviewPass stream deferred bindings");
            if (!result) {
                return result;
            }
            result = bindBuffer(
                *bundle.streamDeferredBindingsBuffer,
                bundle.streamDeferredBindingsHandle,
                "stream deferred bindings");
            if (!result) {
                return result;
            }
            result = uploadStorageBuffer(
                *device_,
                streamOwnerMask_.data(),
                static_cast<uint64_t>(streamOwnerMask_.size()) * sizeof(uint32_t),
                bundle.streamOwnerMaskBuffer,
                log,
                "GPUDrivenPreviewPass stream owner mask");
            if (!result) {
                return result;
            }
            result = bindBuffer(
                *bundle.streamOwnerMaskBuffer,
                bundle.streamOwnerMaskHandle,
                "stream owner mask");
            if (!result) {
                return result;
            }
        }
        result = allocateImage(bundle.depthImageHandle, "depth");
        if (!result) {
            return result;
        }
        result = allocateImage(bundle.visibilityImageHandle, "visibility");
        if (!result) {
            return result;
        }
        result = allocateImage(bundle.cullingDepthImageHandle, "culling depth");
        if (!result) {
            return result;
        }
        result = writeImage(bundle.cullingDepthImageHandle, cullingDepthView, "culling depth");
        if (!result) {
            return result;
        }

        bundle.materialTextureHandles.resize(materialTextures_.size());
        for (size_t textureIndex = 0; textureIndex < materialTextures_.size(); ++textureIndex) {
            GPUDrivenPreviewTextureResource& texture = materialTextures_[textureIndex];
            if (texture.view == nullptr) {
                log = "GPUDrivenPreviewPass material texture view is null";
                return makeError(Error::InvalidArgument);
            }
            result = allocateImage(bundle.materialTextureHandles[textureIndex], "material");
            if (!result) {
                return result;
            }
            result =
                writeImage(bundle.materialTextureHandles[textureIndex], *texture.view, "material");
            if (!result) {
                return result;
            }
        }
        if (bundle.materialTextureHandles.empty()) {
            log = "GPUDrivenPreviewPass material texture descriptor remap is empty";
            return makeError(Error::InvalidArgument);
        }
        if (materialTextureCount_ == 0 ||
            materialTextureCount_ % kGPUSceneMaterialTextureSlotCount != 0) {
            log = "GPUDrivenPreviewPass canonical material texture remap size is invalid";
            return makeError(Error::InvalidArgument);
        }
        std::vector<uint32_t> materialTextureRemap(
            materialTextureCount_,
            bundle.materialTextureHandles.front().index);
        const auto textureForSlot = [](const scene::RenderMaterial& material,
                                       uint32_t slot) -> const scene::RenderTextureInfo* {
            const std::array<const scene::RenderTextureInfo*,
                             kGPUSceneMaterialTextureSlotCount> textures{
                &material.baseColorTexture,
                &material.metallicRoughnessTexture,
                &material.normalTexture,
                &material.occlusionTexture,
                &material.emissiveTexture,
                &material.transmissionTexture,
                &material.thicknessTexture,
                &material.diffuseTransmissionTexture,
                &material.diffuseTransmissionColorTexture,
            };
            return textures[slot];
        };
        std::span<const scene::RenderMaterial> sourceMaterials;
        if (gpuSceneSource_ != nullptr) {
            sourceMaterials = gpuSceneSource_->materials();
        }
        for (uint32_t remapIndex = 0;
             remapIndex < materialTextureRemap.size();
             ++remapIndex) {
            const uint32_t materialIndex =
                remapIndex / kGPUSceneMaterialTextureSlotCount;
            const uint32_t textureSlot =
                remapIndex % kGPUSceneMaterialTextureSlotCount;
            if (materialIndex >= sourceMaterials.size()) {
                continue;
            }
            const int32_t logicalTextureId =
                textureForSlot(sourceMaterials[materialIndex], textureSlot)->textureIndex;
            if (logicalTextureId < 0 ||
                static_cast<size_t>(logicalTextureId) >=
                    logicalTextureToMaterialTexture_.size()) {
                continue;
            }
            const uint32_t consumerTextureIndex =
                logicalTextureToMaterialTexture_[logicalTextureId];
            if (consumerTextureIndex >= bundle.materialTextureHandles.size()) {
                continue;
            }
            materialTextureRemap[remapIndex] =
                bundle.materialTextureHandles[consumerTextureIndex].index;
        }
        result = uploadStorageBuffer(
            *device_, materialTextureRemap.data(),
            static_cast<uint64_t>(materialTextureRemap.size() * sizeof(uint32_t)),
            bundle.materialTextureRemapBuffer, log,
            "GPUDrivenPreviewPass material texture descriptor remap");
        if (!result) {
            return result;
        }
        result = bindBuffer(*bundle.materialTextureRemapBuffer, bundle.materialTextureRemapHandle,
                            "material texture descriptor remap");
        if (!result) {
            return result;
        }

        result = allocateImage(bundle.environmentTextureHandle, "environment");
        if (!result) {
            return result;
        }
        result = bundle.heap->allocateBuffer(bundle.environmentSHBufferHandle);
        if (!result || !bundle.environmentSHBufferHandle.valid()) {
            log += resultMessage("allocateBuffer(GPUDrivenPreviewPass environment SH)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        for (size_t textureIndex = 0; textureIndex < openPBRLut2D_.size(); ++textureIndex) {
            if (openPBRLut2D_[textureIndex].view == nullptr) {
                log = "GPUDrivenPreviewPass OpenPBR 2D LUT view is null";
                return makeError(Error::InvalidArgument);
            }
            result = allocateImage(bundle.openPBRLut2DHandles[textureIndex], "OpenPBR 2D LUT");
            if (!result) {
                return result;
            }
            if (textureIndex > 0 && bundle.openPBRLut2DHandles[textureIndex].index !=
                                        bundle.openPBRLut2DHandles[0].index + textureIndex) {
                log += "GPUDrivenPreviewPass OpenPBR 2D LUT descriptors are not "
                       "contiguous\n";
                return makeError(Error::Failure);
            }
            result = writeImage(bundle.openPBRLut2DHandles[textureIndex],
                                *openPBRLut2D_[textureIndex].view, "OpenPBR 2D LUT");
            if (!result) {
                return result;
            }
        }
        for (size_t textureIndex = 0; textureIndex < openPBRLut3D_.size(); ++textureIndex) {
            if (openPBRLut3D_[textureIndex].view == nullptr) {
                log = "GPUDrivenPreviewPass OpenPBR 3D LUT view is null";
                return makeError(Error::InvalidArgument);
            }
            result = allocateImage(bundle.openPBRLut3DHandles[textureIndex], "OpenPBR 3D LUT");
            if (!result) {
                return result;
            }
            if (textureIndex > 0 && bundle.openPBRLut3DHandles[textureIndex].index !=
                                        bundle.openPBRLut3DHandles[0].index + textureIndex) {
                log += "GPUDrivenPreviewPass OpenPBR 3D LUT descriptors are not "
                       "contiguous\n";
                return makeError(Error::Failure);
            }
            result = writeImage(bundle.openPBRLut3DHandles[textureIndex],
                                *openPBRLut3D_[textureIndex].view, "OpenPBR 3D LUT");
            if (!result) {
                return result;
            }
        }

        if (includeGPUSceneBindings) {
            result = gpuSceneSubsystem_->createBindings(
                *bundle.heap,
                bundle.gpuSceneBindings,
                log);
            if (!result) {
                return result;
            }
        }

        outBundle = std::move(bundle);
        return {};
    }

    void installBindingBundle(GPUDrivenPreviewBindingBundle&& bundle)
    {
        bindlessHeap_ = std::move(bundle.heap);
        materialTextureRemapBuffer_ = std::move(bundle.materialTextureRemapBuffer);
        streamDeferredBindingsBuffer_ =
            std::move(bundle.streamDeferredBindingsBuffer);
        streamOwnerMaskBuffer_ = std::move(bundle.streamOwnerMaskBuffer);
        gpuSceneBindings_ = bundle.gpuSceneBindings;
        materialTextureRemapHandle_ = bundle.materialTextureRemapHandle;
        hzbHandles_ = bundle.hzbHandles;
        deferredColorHandle_ = bundle.deferredColorHandle;
        depthImageHandle_ = bundle.depthImageHandle;
        visibilityImageHandle_ = bundle.visibilityImageHandle;
        cullingDepthImageHandle_ = bundle.cullingDepthImageHandle;
        materialTextureHandles_ = std::move(bundle.materialTextureHandles);
        environmentTextureHandle_ = bundle.environmentTextureHandle;
        environmentSHBufferHandle_ = bundle.environmentSHBufferHandle;
        openPBRLut2DHandles_ = bundle.openPBRLut2DHandles;
        openPBRLut3DHandles_ = bundle.openPBRLut3DHandles;
        streamDeferredResourceHandles_ =
            bundle.streamDeferredResourceHandles;
        streamDeferredBindingsHandle_ =
            bundle.streamDeferredBindingsHandle;
        streamOwnerMaskHandle_ = bundle.streamOwnerMaskHandle;
        for (size_t frameSlot = 0; frameSlot < bundle.frameSlots.size(); ++frameSlot) {
            GPUDrivenPreviewFrameSlotResources& resources = frameSlotResources_[frameSlot];
            const GPUDrivenPreviewFrameSlotBindings& bindings = bundle.frameSlots[frameSlot];
            resources.paramsHandle = bindings.paramsHandle;
            resources.instanceVisibilityHandle = bindings.instanceVisibilityHandle;
            resources.visibleInstanceIdsHandle = bindings.visibleInstanceIdsHandle;
            resources.visibleInstanceCounterHandle = bindings.visibleInstanceCounterHandle;
            resources.visibleMeshletHandles = bindings.visibleMeshletHandles;
            resources.indirectHandles = bindings.indirectHandles;
        }
        bindingViewAllocationId_ = gpuSceneViewAllocationId_;
        environmentBindingValid_ = false;
    }

    Result ensureFrameResources(
        uint32_t width,
        uint32_t height,
        RenderSubsystemHost* subsystemHost)
    {
        width = std::max(width, 1u);
        height = std::max(height, 1u);
        if (frameWidth_ == width &&
            frameHeight_ == height &&
            bindingViewAllocationId_ == gpuSceneViewAllocationId_) {
            return {};
        }
        if (device_ == nullptr || bindlessHeap_ == nullptr || subsystemHost == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const uint32_t mipCount = computeHzbMipCount(width, height);
        const uint64_t elementCount = computeHzbElementCount(width, height, mipCount);
        std::unique_ptr<Buffer> resizedDeferredColorBuffer;
        GPUDrivenPreviewCullingTargets resizedCullingTargets;
        std::string log;
        Result result = createDeviceStorageBuffer(
            *device_,
            static_cast<uint64_t>(width) * height * sizeof(uint32_t),
            BufferUsageBits::Storage,
            resizedDeferredColorBuffer,
            log,
            "resized deferred color");
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
            return result;
        }
        result = createCullingTargets(
            *device_,
            width,
            height,
            resizedCullingTargets,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
            return result;
        }

        const uint32_t previousHzbMipCount = hzbMipCount_;
        const uint64_t previousHzbElementCount = hzbElementCount_;
        hzbMipCount_ = mipCount;
        hzbElementCount_ = elementCount;
        result = ensureGPUSceneViewResources(width, height, log);
        if (!result) {
            hzbMipCount_ = previousHzbMipCount;
            hzbElementCount_ = previousHzbElementCount;
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
            return result;
        }

        GPUDrivenPreviewBindingBundle resizedBindings;
        result = createBindingBundle(
            *resizedDeferredColorBuffer,
            *resizedCullingTargets.depthView,
            true,
            resizedBindings,
            log);
        if (!result) {
            hzbMipCount_ = previousHzbMipCount;
            hzbElementCount_ = previousHzbElementCount;
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
            return result;
        }

        auto retired = std::make_shared<GPUDrivenPreviewRetiredViewResources>();
        retired->deferredColorBuffer = std::move(deferredColorBuffer_);
        retired->cullingTargets = std::move(cullingTargets_);
        retired->bindlessHeap = std::move(bindlessHeap_);
        retired->materialTextureRemapBuffer = std::move(materialTextureRemapBuffer_);
        retired->streamDeferredBindingsBuffer =
            std::move(streamDeferredBindingsBuffer_);
        retired->streamOwnerMaskBuffer = std::move(streamOwnerMaskBuffer_);
        deferredColorBuffer_ = std::move(resizedDeferredColorBuffer);
        cullingTargets_ = std::move(resizedCullingTargets);
        installBindingBundle(std::move(resizedBindings));
        subsystemHost->retire(std::static_pointer_cast<void>(retired));

        spdlog::info(
            "[GPUDrivenPreviewPass] Resized frame resources {}x{} -> {}x{}",
            frameWidth_,
            frameHeight_,
            width,
            height);
        frameWidth_ = width;
        frameHeight_ = height;
        frameIndex_ = 0;
        invalidateHzbHistory();
        previousCameraValid_ = false;
        frameBuffersInitialized_ = false;
        cullingTargetsInitialized_ = false;
        return {};
    }

    static std::filesystem::path scenePathFromProperties(const RenderGraphProperties& props)
    {
        if (props.contains("path") && props["path"].is_string()) {
            std::filesystem::path path = props["path"].get<std::string>();
            if (path.is_relative()) {
                path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
            }
            return path;
        }
        return kDefaultGPUDrivenScenePath;
    }

    static float finiteOr(float value, float fallback)
    {
        return std::isfinite(value) ? value : fallback;
    }

    static const RenderGraphProperties* cameraPropertiesFrom(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return nullptr;
        }
        auto iter = properties.find("camera");
        if (iter == properties.end() || !iter->is_object()) {
            return nullptr;
        }
        return &(*iter);
    }

    static float cameraFloat(
        const RenderGraphProperties* camera,
        const char* key,
        float fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_number()) {
            return fallback;
        }
        return finiteOr(iter->get<float>(), fallback);
    }

    static float3 cameraVec3(
        const RenderGraphProperties* camera,
        const char* key,
        const float3& fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_array() || iter->size() < 3) {
            return fallback;
        }

        float values[3] = {fallback.x, fallback.y, fallback.z};
        for (size_t index = 0; index < 3; ++index) {
            const RenderGraphProperties& component = (*iter)[index];
            if (component.is_number()) {
                values[index] = finiteOr(component.get<float>(), values[index]);
            }
        }
        return float3(values[0], values[1], values[2]);
    }

    static bool cameraIsOrthographic(const RenderGraphProperties* camera)
    {
        if (camera == nullptr) {
            return false;
        }
        auto iter = camera->find("projection");
        if (iter == camera->end() || !iter->is_string()) {
            return false;
        }
        const std::string projection = iter->get<std::string>();
        return projection == "orthographic" || projection == "ortho";
    }

    static uint32_t previewModeFromProperties(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return kGPUDrivenPreviewModeMeshlet;
        }
        auto iter = properties.find("mode");
        if (iter == properties.end() || !iter->is_string()) {
            return kGPUDrivenPreviewModeMeshlet;
        }
        const std::string value = iter->get<std::string>();
        if (value == "primitive" || value == "perPrimitive" || value == "per primitive") {
            return kGPUDrivenPreviewModePrimitive;
        }
        if (value == "lod" || value == "lodLevel" || value == "lod level" || value == "LOD") {
            return kGPUDrivenPreviewModeLod;
        }
        if (value == "shaded" || value == "openpbr" || value == "material") {
            return kGPUDrivenPreviewModeShaded;
        }
        if (value == "baseColor" || value == "base color" || value == "albedo") {
            return kGPUDrivenPreviewModeBaseColor;
        }
        return kGPUDrivenPreviewModeMeshlet;
    }

    static uint32_t lodLevelFromProperties(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return 0;
        }
        auto iter = properties.find("lodLevel");
        if (iter == properties.end() || !iter->is_number_integer()) {
            return 0;
        }
        return static_cast<uint32_t>(std::clamp(iter->get<int32_t>(), 0, 31));
    }

    static GPUDrivenPreviewMeshletRange selectedMeshletRange(
        uint32_t mode,
        uint32_t requestedLodLevel,
        const GPUDrivenPreviewMeshletRange& baseRange,
        const std::vector<GPUDrivenPreviewMeshletRange>& lodLevelRanges,
        uint32_t& outSelectedLodLevel)
    {
        outSelectedLodLevel = 0;
        if (mode != kGPUDrivenPreviewModeLod || lodLevelRanges.empty()) {
            return baseRange;
        }

        uint32_t lodLevel = std::min<uint32_t>(
            requestedLodLevel,
            static_cast<uint32_t>(lodLevelRanges.size() - 1u));
        if (lodLevelRanges[lodLevel].count == 0) {
            uint32_t fallback = lodLevel;
            while (fallback > 0 && lodLevelRanges[fallback].count == 0) {
                --fallback;
            }
            if (lodLevelRanges[fallback].count == 0) {
                for (uint32_t index = lodLevel + 1u; index < lodLevelRanges.size(); ++index) {
                    if (lodLevelRanges[index].count != 0) {
                        fallback = index;
                        break;
                    }
                }
            }
            lodLevel = fallback;
        }

        if (lodLevelRanges[lodLevel].count == 0) {
            return baseRange;
        }
        outSelectedLodLevel = lodLevel;
        return lodLevelRanges[lodLevel];
    }

    static uint32_t maxMeshletRangeCount(
        const GPUDrivenPreviewMeshletRange& baseRange,
        const std::vector<GPUDrivenPreviewMeshletRange>& lodLevelRanges)
    {
        uint32_t result = baseRange.count;
        for (const GPUDrivenPreviewMeshletRange& range : lodLevelRanges) {
            result = std::max(result, range.count);
        }
        return result;
    }

    static void writeParamVec3(const float3& value, float out[4], float w)
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = w;
    }

    static void copyCullingCamera(
        const GPUDrivenPreviewGpuParams& source,
        GPUDrivenPreviewGpuParams& destination)
    {
        std::memcpy(destination.eye, source.eye, sizeof(destination.eye));
        std::memcpy(destination.center, source.center, sizeof(destination.center));
        std::memcpy(destination.upProjection, source.upProjection, sizeof(destination.upProjection));
        std::memcpy(destination.viewport, source.viewport, sizeof(destination.viewport));
        std::memcpy(destination.clipOrtho, source.clipOrtho, sizeof(destination.clipOrtho));
    }

    static void copyCullingCameraToPrevious(
        const GPUDrivenPreviewGpuParams& source,
        GPUDrivenPreviewGpuParams& destination)
    {
        std::memcpy(destination.previousEye, source.eye, sizeof(destination.previousEye));
        std::memcpy(destination.previousCenter, source.center, sizeof(destination.previousCenter));
        std::memcpy(destination.previousUpProjection, source.upProjection, sizeof(destination.previousUpProjection));
        std::memcpy(destination.previousViewport, source.viewport, sizeof(destination.previousViewport));
        std::memcpy(destination.previousClipOrtho, source.clipOrtho, sizeof(destination.previousClipOrtho));
    }

    static void copyCullingCameraToRender(GPUDrivenPreviewGpuParams& params)
    {
        std::memcpy(params.renderEye, params.eye, sizeof(params.renderEye));
        std::memcpy(params.renderCenter, params.center, sizeof(params.renderCenter));
        std::memcpy(params.renderUpProjection, params.upProjection, sizeof(params.renderUpProjection));
        std::memcpy(params.renderViewport, params.viewport, sizeof(params.renderViewport));
        std::memcpy(params.renderClipOrtho, params.clipOrtho, sizeof(params.renderClipOrtho));
    }

    struct PrimitiveInstanceRef {
        uint32_t geometryIndex = 0;
        uint32_t primitiveIndex = 0;
        uint32_t materialIndex = 0;
        uint32_t transformIndex = 0;
        uint32_t instanceIndex = 0;
        uint32_t meshletFlags = 0;
    };

    static bool appendPrimitiveVertices(
        const scene::RenderPrimitive& primitive,
        std::vector<GPUDrivenPreviewGpuVertex>& outVertices,
        uint32_t& outPositionBase,
        std::string& log)
    {
        if (outVertices.size() + primitive.positions.size() > std::numeric_limits<uint32_t>::max()) {
            log = "GPUDrivenPreviewPass scene is too large to address with uint32 vertex indices";
            return false;
        }

        outPositionBase = static_cast<uint32_t>(outVertices.size());
        outVertices.reserve(outVertices.size() + primitive.positions.size());
        for (size_t vertexIndex = 0; vertexIndex < primitive.positions.size(); ++vertexIndex) {
            const float3& localPosition = primitive.positions[vertexIndex];
            const float3 localNormal = vertexIndex < primitive.normals.size()
                ? primitive.normals[vertexIndex]
                : float3(0.0f, 0.0f, 1.0f);
            const float4 localTangent = vertexIndex < primitive.tangents.size()
                ? primitive.tangents[vertexIndex]
                : float4(1.0f, 0.0f, 0.0f, 1.0f);
            const float2 texcoord = vertexIndex < primitive.texcoords0.size()
                ? primitive.texcoords0[vertexIndex]
                : float2(0.0f, 0.0f);
            GPUDrivenPreviewGpuVertex vertex;
            vertex.position[0] = localPosition.x;
            vertex.position[1] = localPosition.y;
            vertex.position[2] = localPosition.z;
            vertex.normal[0] = localNormal.x;
            vertex.normal[1] = localNormal.y;
            vertex.normal[2] = localNormal.z;
            vertex.tangent[0] = localTangent.x;
            vertex.tangent[1] = localTangent.y;
            vertex.tangent[2] = localTangent.z;
            vertex.tangent[3] = localTangent.w;
            vertex.texcoord[0] = texcoord.x;
            vertex.texcoord[1] = texcoord.y;
            outVertices.push_back(vertex);
        }
        return true;
    }

    static bool appendPrimitiveClusters(
        const scene::RenderPrimitive& primitive,
        const std::vector<scene::MeshletCluster>& clusters,
        const std::vector<uint32_t>& clusterVertices,
        const std::vector<uint8_t>& clusterTriangles,
        uint32_t firstCluster,
        uint32_t clusterCount,
        uint32_t positionBase,
        std::vector<GPUDrivenPreviewGpuMeshlet>& outMeshlets,
        std::vector<uint32_t>& outMeshletVertices,
        std::vector<uint32_t>& outMeshletTriangles,
        std::string& log)
    {
        if (static_cast<size_t>(firstCluster) + clusterCount > clusters.size()) {
            log = "GPUDrivenPreviewPass found invalid meshlet cluster range";
            return false;
        }

        for (uint32_t clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex) {
            const scene::MeshletCluster& cluster = clusters[static_cast<size_t>(firstCluster) + clusterIndex];
            if (cluster.vertexCount == 0 ||
                cluster.triangleCount == 0 ||
                cluster.vertexCount > 128 ||
                cluster.triangleCount > 128 ||
                static_cast<size_t>(cluster.vertexOffset) + cluster.vertexCount > clusterVertices.size() ||
                static_cast<size_t>(cluster.triangleOffset) + static_cast<size_t>(cluster.triangleCount) * 3u >
                    clusterTriangles.size()) {
                log = "GPUDrivenPreviewPass found invalid meshlet cluster data";
                return false;
            }

            const uint32_t meshletVertexOffset = static_cast<uint32_t>(outMeshletVertices.size());
            const uint32_t meshletTriangleOffset = static_cast<uint32_t>(outMeshletTriangles.size());

            for (uint32_t vertexIndex = 0; vertexIndex < cluster.vertexCount; ++vertexIndex) {
                const uint32_t localVertex =
                    clusterVertices[static_cast<size_t>(cluster.vertexOffset) + vertexIndex];
                if (localVertex >= primitive.positions.size()) {
                    log = "GPUDrivenPreviewPass found out-of-range meshlet vertex reference";
                    return false;
                }
                outMeshletVertices.push_back(positionBase + localVertex);
            }

            for (uint32_t triangleIndex = 0; triangleIndex < cluster.triangleCount * 3u; ++triangleIndex) {
                const uint32_t localVertex =
                    clusterTriangles[static_cast<size_t>(cluster.triangleOffset) + triangleIndex];
                if (localVertex >= cluster.vertexCount) {
                    log = "GPUDrivenPreviewPass found out-of-range meshlet triangle index";
                    return false;
                }
                outMeshletTriangles.push_back(localVertex);
            }

            outMeshlets.push_back(GPUDrivenPreviewGpuMeshlet{
                .vertexOffset = meshletVertexOffset,
                .vertexCount = cluster.vertexCount,
                .triangleOffset = meshletTriangleOffset,
                .triangleCount = cluster.triangleCount,
                .lodLevel = cluster.lodLevel,
                .lodGroupIndex = static_cast<uint32_t>(std::max(cluster.lodGroupIndex, 0)),
                .boundingSphere = {
                    cluster.boundingSphereCenter.x,
                    cluster.boundingSphereCenter.y,
                    cluster.boundingSphereCenter.z,
                    std::max(cluster.boundingSphereRadius, 0.0f),
                },
                .coneApexCutoff = {
                    cluster.coneApex.x,
                    cluster.coneApex.y,
                    cluster.coneApex.z,
                    cluster.coneCutoff,
                },
                .coneAxis = {
                    cluster.coneAxis.x,
                    cluster.coneAxis.y,
                    cluster.coneAxis.z,
                    0.0f,
                },
            });
        }

        return true;
    }

    static bool loadMeshletScene(
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        std::vector<GPUDrivenPreviewGpuVertex>& outVertices,
        std::vector<GPUDrivenPreviewGpuMeshlet>& outMeshlets,
        std::vector<GPUDrivenPreviewGpuMeshletDraw>& outMeshletDraws,
        std::vector<uint32_t>& outMeshletVertices,
        std::vector<uint32_t>& outMeshletTriangles,
        std::vector<SceneGpuTransform>& outTransforms,
        std::vector<GPUDrivenPreviewGpuInstance>& outInstances,
        std::vector<uint32_t>& outInstanceRenderNodeIndices,
        scene::Bounds& outBounds,
        GPUDrivenPreviewMeshletRange& outBaseMeshletRange,
        std::vector<GPUDrivenPreviewMeshletRange>& outLodLevelRanges,
        std::string& log)
    {
        const std::filesystem::path path = scenePathFromProperties(properties);
        if (runtimeScene == nullptr) {
            log = "GPUDrivenPreviewPass requires a runtime scene resource provider";
            return false;
        }
        const scene::Scene& loadedScene = *runtimeScene;
        if (!loadedScene.bounds().valid) {
            log = "GPUDrivenPreviewPass scene bounds are unavailable";
            return false;
        }

        outVertices.clear();
        outMeshlets.clear();
        outMeshletDraws.clear();
        outMeshletVertices.clear();
        outMeshletTriangles.clear();
        outInstances.clear();
        outInstanceRenderNodeIndices.clear();
        outBaseMeshletRange = GPUDrivenPreviewMeshletRange{};
        outLodLevelRanges.clear();
        outBounds = loadedScene.bounds();
        outTransforms = buildSceneGpuTransforms(loadedScene);

        struct DrawableSource {
            const scene::RenderPrimitive* primitive = nullptr;
            const scene::RenderNode* renderNode = nullptr;
            uint32_t renderNodeIndex = 0;
            uint32_t renderPrimitiveIndex = 0;
        };
        std::vector<DrawableSource> drawableSources;
        std::vector<GPUDrivenPreviewGeometrySource> geometrySources;
        drawableSources.reserve(loadedScene.renderNodes().size());
        geometrySources.reserve(loadedScene.renderNodes().size());
        const auto resolveMaterialIndex = [&](const scene::RenderNode& node,
                                              const scene::RenderPrimitive& primitive) {
            const int32_t materialIndex = node.materialIndex >= 0
                ? node.materialIndex
                : primitive.materialIndex;
            return materialIndex >= 0 &&
                    static_cast<size_t>(materialIndex) < loadedScene.materials().size()
                ? static_cast<uint32_t>(materialIndex)
                : 0u;
        };
        for (size_t renderNodeIndex = 0; renderNodeIndex < loadedScene.renderNodes().size(); ++renderNodeIndex) {
            const scene::RenderNode& renderNode = loadedScene.renderNodes()[renderNodeIndex];
            if (renderNode.renderPrimitiveIndex < 0 ||
                static_cast<size_t>(renderNode.renderPrimitiveIndex) >= loadedScene.renderPrimitives().size()) {
                continue;
            }

            const scene::RenderPrimitive& primitive =
                loadedScene.renderPrimitives()[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
            if (primitive.mode != kGltfTriangleListMode ||
                primitive.positions.empty() ||
                primitive.meshletClusters.empty()) {
                continue;
            }
            if (!primitive.localBounds.valid ||
                drawableSources.size() >= std::numeric_limits<uint32_t>::max()) {
                log = "GPUDrivenPreviewPass found invalid or unaddressable instance bounds";
                return false;
            }

            drawableSources.push_back(DrawableSource{
                .primitive = &primitive,
                .renderNode = &renderNode,
                .renderNodeIndex = static_cast<uint32_t>(renderNodeIndex),
                .renderPrimitiveIndex = static_cast<uint32_t>(renderNode.renderPrimitiveIndex),
            });
            geometrySources.push_back(GPUDrivenPreviewGeometrySource{
                .primitive = &primitive,
                .renderPrimitiveIndex = static_cast<uint32_t>(renderNode.renderPrimitiveIndex),
            });
        }

        const GPUDrivenPreviewGeometryDedupPlan geometryPlan =
            buildGPUDrivenPreviewGeometryDedupPlan(geometrySources);
        if (geometryPlan.conflictingPayloadCount != 0) {
            spdlog::warn(
                "[GPUDrivenPreviewPass] {} geometry key payload conflict(s); preserving independent payloads",
                geometryPlan.conflictingPayloadCount);
        }

        struct GeometryPayload {
            const scene::RenderPrimitive* primitive = nullptr;
            uint32_t positionBase = 0;
            GPUDrivenPreviewMeshletRange baseRange;
            std::vector<GPUDrivenPreviewMeshletRange> lodRanges;
        };
        std::vector<GeometryPayload> geometryPayloads(geometryPlan.geometryCount);
        std::vector<PrimitiveInstanceRef> primitiveInstances;
        primitiveInstances.reserve(drawableSources.size());
        size_t maxLodLevelCount = 0;
        for (size_t sourceIndex = 0; sourceIndex < drawableSources.size(); ++sourceIndex) {
            const DrawableSource& source = drawableSources[sourceIndex];
            const scene::RenderPrimitive& primitive = *source.primitive;
            const scene::RenderNode& renderNode = *source.renderNode;
            const uint32_t geometryIndex = geometryPlan.geometryIndices[sourceIndex];
            if (geometryIndex >= geometryPayloads.size()) {
                log = "GPUDrivenPreviewPass produced an invalid geometry dedup assignment";
                return false;
            }
            GeometryPayload& geometry = geometryPayloads[geometryIndex];
            if (geometry.primitive == nullptr) {
                geometry.primitive = &primitive;
                if (!appendPrimitiveVertices(primitive, outVertices, geometry.positionBase, log)) {
                    return false;
                }

                geometry.baseRange.offset = static_cast<uint32_t>(outMeshlets.size());
                if (!appendPrimitiveClusters(
                        primitive,
                        primitive.meshletClusters,
                        primitive.meshletVertices,
                        primitive.meshletTriangles,
                        0,
                        static_cast<uint32_t>(primitive.meshletClusters.size()),
                        geometry.positionBase,
                        outMeshlets,
                        outMeshletVertices,
                        outMeshletTriangles,
                        log)) {
                    return false;
                }
                geometry.baseRange.count =
                    static_cast<uint32_t>(outMeshlets.size()) - geometry.baseRange.offset;

                geometry.lodRanges.resize(primitive.meshletLodLevels.size());
                for (uint32_t lodLevel = 0;
                     lodLevel < primitive.meshletLodLevels.size();
                     ++lodLevel) {
                    const scene::MeshletLodLevel& level = primitive.meshletLodLevels[lodLevel];
                    GPUDrivenPreviewMeshletRange& lodRange = geometry.lodRanges[lodLevel];
                    lodRange.offset = static_cast<uint32_t>(outMeshlets.size());
                    if (!appendPrimitiveClusters(
                            primitive,
                            primitive.meshletLodClusters,
                            primitive.meshletLodVertices,
                            primitive.meshletLodTriangles,
                            level.clusterOffset,
                            level.clusterCount,
                            geometry.positionBase,
                            outMeshlets,
                            outMeshletVertices,
                            outMeshletTriangles,
                            log)) {
                        return false;
                    }
                    lodRange.count = static_cast<uint32_t>(outMeshlets.size()) - lodRange.offset;
                }
            }

            const uint32_t materialIndex = resolveMaterialIndex(renderNode, primitive);
            const uint32_t instanceIndex = static_cast<uint32_t>(outInstances.size());
            const float3 instanceCenter = primitive.localBounds.center();
            outInstances.push_back(GPUDrivenPreviewGpuInstance{
                .boundingSphere = {
                    instanceCenter.x,
                    instanceCenter.y,
                    instanceCenter.z,
                    std::max(primitive.localBounds.radius(), 0.000001f),
                },
                .transformIndex = source.renderNodeIndex,
                .primitiveIndex = source.renderPrimitiveIndex,
                .flags = renderNode.visible ? kGPUDrivenPreviewInstanceVisible : 0u,
            });
            outInstanceRenderNodeIndices.push_back(source.renderNodeIndex);
            const scene::RenderMaterial* material =
                materialIndex < loadedScene.materials().size()
                ? &loadedScene.materials()[materialIndex]
                : nullptr;
            uint32_t meshletFlags = 0;
            if (material != nullptr && material->doubleSided) {
                meshletFlags |= kGPUDrivenPreviewMeshletDoubleSided;
            }
            if (material != nullptr && material->alphaMode == "MASK") {
                meshletFlags |= kGPUDrivenPreviewMeshletAlphaMasked;
            } else if (material != nullptr && material->alphaMode == "BLEND") {
                meshletFlags |= kGPUDrivenPreviewMeshletAlphaBlend;
            }
            primitiveInstances.push_back(PrimitiveInstanceRef{
                .geometryIndex = geometryIndex,
                .primitiveIndex = source.renderPrimitiveIndex,
                .materialIndex = materialIndex,
                .transformIndex = source.renderNodeIndex,
                .instanceIndex = instanceIndex,
                .meshletFlags = meshletFlags,
            });
            maxLodLevelCount = std::max(maxLodLevelCount, primitive.meshletLodLevels.size());
        }

        const auto appendDrawRange = [&](const PrimitiveInstanceRef& instance,
                                         const GPUDrivenPreviewMeshletRange& geometryRange) {
            if (static_cast<uint64_t>(outMeshletDraws.size()) + geometryRange.count >
                std::numeric_limits<uint32_t>::max()) {
                log = "GPUDrivenPreviewPass scene is too large to address meshlet draws";
                return false;
            }
            for (uint32_t meshletOffset = 0; meshletOffset < geometryRange.count; ++meshletOffset) {
                outMeshletDraws.push_back(GPUDrivenPreviewGpuMeshletDraw{
                    .geometryMeshletIndex = geometryRange.offset + meshletOffset,
                    .primitiveIndex = instance.primitiveIndex,
                    .materialIndex = instance.materialIndex,
                    .transformIndex = instance.transformIndex,
                    .instanceIndex = instance.instanceIndex,
                    .flags = instance.meshletFlags,
                });
            }
            return true;
        };

        outBaseMeshletRange.offset = static_cast<uint32_t>(outMeshletDraws.size());
        for (const PrimitiveInstanceRef& instance : primitiveInstances) {
            if (!appendDrawRange(instance, geometryPayloads[instance.geometryIndex].baseRange)) {
                return false;
            }
        }
        outBaseMeshletRange.count =
            static_cast<uint32_t>(outMeshletDraws.size()) - outBaseMeshletRange.offset;

        outLodLevelRanges.resize(maxLodLevelCount);
        for (uint32_t lodLevel = 0; lodLevel < maxLodLevelCount; ++lodLevel) {
            GPUDrivenPreviewMeshletRange range;
            range.offset = static_cast<uint32_t>(outMeshletDraws.size());

            for (const PrimitiveInstanceRef& instance : primitiveInstances) {
                const GeometryPayload& geometry = geometryPayloads[instance.geometryIndex];
                if (lodLevel >= geometry.lodRanges.size()) {
                    continue;
                }
                if (!appendDrawRange(instance, geometry.lodRanges[lodLevel])) {
                    return false;
                }
            }

            range.count = static_cast<uint32_t>(outMeshletDraws.size()) - range.offset;
            outLodLevelRanges[lodLevel] = range;
        }

        if (outVertices.empty() ||
            outMeshlets.empty() ||
            outMeshletDraws.empty() ||
            outMeshletVertices.empty() ||
            outMeshletTriangles.empty() ||
            outInstances.empty()) {
            log = "GPUDrivenPreviewPass found no drawable meshlet geometry in " + path.string();
            return false;
        }
        if (!visibilityRecordCapacityFitsId(outMeshletDraws.size())) {
            log = "GPUDrivenPreviewPass scene has too many meshlets for the packed visibility format";
            return false;
        }
        if (outBaseMeshletRange.count == 0) {
            log = "GPUDrivenPreviewPass found no base meshlet geometry in " + path.string();
            return false;
        }
        return true;
    }

    static void buildParams(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        const scene::Bounds& drawBounds,
        const GPUDrivenPreviewMeshletRange& baseMeshletRange,
        const std::vector<GPUDrivenPreviewMeshletRange>& lodLevelRanges,
        uint32_t instanceCount,
        uint32_t hzbMipCount,
        uint32_t frameIndex,
        bool hzbValid,
        const GPUDrivenPreviewGpuParams* previousParams,
        uint32_t materialTextureCount,
        uint32_t materialCount,
        const EnvironmentSettings& environment,
        bool environmentMapAvailable,
        GPUDrivenPreviewGpuParams& outParams)
    {
        outParams = GPUDrivenPreviewGpuParams{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 60.0f),
            1.0f,
            179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + radius
                : radius * 2.5f,
            0.05f);
        const float3 defaultEye(center.x, center.y, center.z + defaultDistance);
        const float3 eye = cameraVec3(cameraProperties, "eye", defaultEye);
        const float3 target = cameraVec3(cameraProperties, "center", center);
        const float3 up = cameraVec3(cameraProperties, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.1f), 0.0001f);
        const float zFar = std::max(
            cameraFloat(cameraProperties, "zfar", defaultDistance + radius * 3.0f),
            zNear + 0.001f);
        const float cameraDistance = std::max(length(eye - target), 0.001f);
        const float defaultOrthoHeight = std::max(2.0f * cameraDistance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float orthoHeight = std::max(
            cameraFloat(cameraProperties, "orthoHeight", defaultOrthoHeight),
            0.0001f);

        writeParamVec3(eye, outParams.eye, 0.0f);
        writeParamVec3(target, outParams.center, 0.0f);
        writeParamVec3(up, outParams.upProjection, cameraIsOrthographic(cameraProperties) ? 1.0f : 0.0f);
        outParams.viewport[0] = aspect;
        outParams.viewport[1] = static_cast<float>(width);
        outParams.viewport[2] = static_cast<float>(height);
        outParams.viewport[3] = fovRadians;
        outParams.clipOrtho[0] = zNear;
        outParams.clipOrtho[1] = zFar;
        outParams.clipOrtho[2] = orthoHeight;
        outParams.clipOrtho[3] = kDefaultReversedZ ? 1.0f : 0.0f;
        copyCullingCameraToRender(outParams);
        outParams.clearColor[0] = 0.015f;
        outParams.clearColor[1] = 0.018f;
        outParams.clearColor[2] = 0.024f;
        outParams.clearColor[3] = 1.0f;
        const uint32_t mode = previewModeFromProperties(properties);
        uint32_t selectedLodLevel = 0;
        const GPUDrivenPreviewMeshletRange meshletRange = selectedMeshletRange(
            mode,
            lodLevelFromProperties(properties),
            baseMeshletRange,
            lodLevelRanges,
            selectedLodLevel);
        outParams.mode = mode;
        outParams.meshletOffset = meshletRange.offset;
        outParams.meshletCount = meshletRange.count;
        outParams.selectedLodLevel = selectedLodLevel;
        outParams.instanceCount = instanceCount;
        outParams.width = std::max(width, 1u);
        outParams.height = std::max(height, 1u);
        outParams.hzbMipCount = std::max(hzbMipCount, 1u);
        outParams.frameIndex = frameIndex;
        outParams.hzbValid = hzbValid ? 1u : 0u;
        outParams.cullingFlags = 0;
        if (boolProperty(&properties, "instanceFrustumCull", true)) {
            outParams.cullingFlags |= kGPUDrivenPreviewCullInstanceFrustum;
        }
        if (boolProperty(&properties, "instanceHzbCull", true)) {
            outParams.cullingFlags |= kGPUDrivenPreviewCullInstanceHzb;
        }
        if (boolProperty(&properties, "meshletFrustumCull", true)) {
            outParams.cullingFlags |= kGPUDrivenPreviewCullMeshletFrustum;
        }
        if (boolProperty(&properties, "meshletNormalConeCull", true)) {
            outParams.cullingFlags |= kGPUDrivenPreviewCullMeshletNormalCone;
        }
        outParams.materialTextureCount = std::max(materialTextureCount, 1u);
        outParams.materialCount = std::max(materialCount, 1u);
        outParams.visibleMeshletCapacity = 1u;
        outParams.environmentIntensity = std::max(environment.intensity, 0.0f);
        outParams.environmentRotationRadians =
            environment.rotationDegrees * (kPi / 180.0f);
        outParams.environmentMode = !environment.enabled
            ? kGPUDrivenEnvironmentModeDisabled
            : (environmentMapAvailable
                ? kGPUDrivenEnvironmentModeMap
                : kGPUDrivenEnvironmentModeProcedural);
        outParams.environmentVisible =
            environment.enabled && environment.visible ? 1u : 0u;
        const GPUDrivenPreviewGpuParams& previous = previousParams != nullptr ? *previousParams : outParams;
        std::memcpy(outParams.previousEye, previous.eye, sizeof(outParams.previousEye));
        std::memcpy(outParams.previousCenter, previous.center, sizeof(outParams.previousCenter));
        std::memcpy(outParams.previousUpProjection, previous.upProjection, sizeof(outParams.previousUpProjection));
        std::memcpy(outParams.previousViewport, previous.viewport, sizeof(outParams.previousViewport));
        std::memcpy(outParams.previousClipOrtho, previous.clipOrtho, sizeof(outParams.previousClipOrtho));
    }

    Result updateParamsBuffer(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        const EnvironmentSettings& environment,
        bool environmentMapAvailable)
    {
        GPUDrivenPreviewFrameSlotResources& slot = activeFrameResources();
        if (slot.paramsBuffer == nullptr || !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }

        const bool freezeCullingCamera = boolProperty(
            &properties,
            "freezeCullingCamera",
            false);
        const bool freezeStateChanged = freezeCullingCamera != freezeCullingCamera_;
        if (freezeStateChanged) {
            frameIndex_ = 0;
            invalidateHzbHistory();
            previousCameraValid_ = false;
            if (!freezeCullingCamera) {
                frozenCullingCameraValid_ = false;
            }
        }

        GPUDrivenPreviewGpuParams params;
        buildParams(
            width,
            height,
            properties,
            drawBounds_,
            baseMeshletRange_,
            lodLevelRanges_,
            instanceCount_,
            hzbMipCount_,
            frameIndex_,
            hzbValid_,
            previousCameraValid_ ? &previousParams_ : nullptr,
            materialTextureCount_,
            materialCount_,
            environment,
            environmentMapAvailable,
            params);

        if (freezeCullingCamera &&
            (!frozenCullingCameraValid_ || freezeStateChanged)) {
            frozenCullingCamera_ = params;
            frozenCullingCameraValid_ = true;
        }
        if (freezeCullingCamera) {
            copyCullingCamera(frozenCullingCamera_, params);
        }

        const GPUDrivenPreviewGpuParams& previousCullingCamera =
            previousCameraValid_ ? previousParams_ : params;
        copyCullingCameraToPrevious(previousCullingCamera, params);

        void* mapped = slot.paramsBuffer->map();
        if (mapped == nullptr) {
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, &params, sizeof(params));
        slot.paramsBuffer->flush(0, sizeof(params));
        slot.paramsBuffer->unmap();
        activeMeshletCount_ = params.meshletCount;
        previousParams_ = params;
        previousCameraValid_ = true;
        freezeCullingCamera_ = freezeCullingCamera;
        return {};
    }

    std::unique_ptr<Buffer> materialTextureRemapBuffer_;
    std::unique_ptr<Buffer> streamDeferredBindingsBuffer_;
    std::unique_ptr<Buffer> streamOwnerMaskBuffer_;
    std::vector<GPUDrivenPreviewFrameSlotResources> frameSlotResources_;
    std::array<Buffer*, 2> hzbBuffers_{};
    std::unique_ptr<Buffer> deferredColorBuffer_;
    GPUDrivenPreviewCullingTargets cullingTargets_;
    MeshletStreamRuntime streamRuntime_;
    std::vector<GPUDrivenPreviewTextureResource> materialTextures_;
    std::array<GPUDrivenPreviewTextureResource, kGPUDrivenOpenPBRLut2DCount> openPBRLut2D_;
    std::array<GPUDrivenPreviewTextureResource, kGPUDrivenOpenPBRLut3DCount> openPBRLut3D_;
    Device* device_ = nullptr;
    GPUSceneSubsystem* gpuSceneSubsystem_ = nullptr;
    const scene::Scene* gpuSceneSource_ = nullptr;
    const scene::Scene* compiledScene_ = nullptr;
    GPUSceneSourceOverrideToken gpuSceneSourceToken_;
    GPUSceneViewId gpuSceneView_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    GPUSceneConsumerBindings gpuSceneBindings_;
    BindlessHandle materialTextureRemapHandle_;
    std::array<BindlessHandle, 2> hzbHandles_;
    BindlessHandle deferredColorHandle_;
    BindlessHandle depthImageHandle_;
    BindlessHandle visibilityImageHandle_;
    BindlessHandle cullingDepthImageHandle_;
    std::vector<BindlessHandle> materialTextureHandles_;
    BindlessHandle environmentTextureHandle_;
    BindlessHandle environmentSHBufferHandle_;
    std::array<BindlessHandle, kGPUDrivenOpenPBRLut2DCount> openPBRLut2DHandles_;
    std::array<BindlessHandle, kGPUDrivenOpenPBRLut3DCount> openPBRLut3DHandles_;
    std::array<BindlessHandle, GPUDrivenStreamDeferredResourceCount>
        streamDeferredResourceHandles_;
    BindlessHandle streamDeferredBindingsHandle_;
    BindlessHandle streamOwnerMaskHandle_;
    BindlessHandle streamVisibilityImageHandle_;
    BindlessHandle streamDepthImageHandle_;
    BindlessHandle streamInstanceVisibilityHandle_;
    BindlessHandle streamVisibleInstanceIdsHandle_;
    BindlessHandle streamVisibleInstanceCounterHandle_;
    std::array<BindlessHandle, 2> streamHzbHandles_;
    std::unique_ptr<ShaderModule> amplificationShader_;
    std::unique_ptr<ShaderModule> meshShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<ShaderModule> maskedFragmentShader_;
    std::unique_ptr<ShaderModule> resetShader_;
    std::unique_ptr<ShaderModule> instanceCullShader_;
    std::unique_ptr<ShaderModule> hzbShader_;
    std::unique_ptr<ShaderModule> deferredShader_;
    std::unique_ptr<ShaderModule> compositeVertexShader_;
    std::unique_ptr<ShaderModule> compositeFragmentShader_;
    std::unique_ptr<ShaderModule> streamMeshShader_;
    std::unique_ptr<ShaderModule> streamFragmentShader_;
    std::unique_ptr<ShaderModule> streamCullResetShader_;
    std::unique_ptr<ShaderModule> streamInstanceCullShader_;
    std::unique_ptr<PipelineCache> pipelineCache_;
    std::array<std::unique_ptr<GraphicsPipeline>, kGPUDrivenPreviewDrawBucketCount> visibilityPipelines_;
    std::unique_ptr<GraphicsPipeline> compositePipeline_;
    std::unique_ptr<ComputePipeline> resetPipeline_;
    std::unique_ptr<ComputePipeline> instanceCullPipeline_;
    std::unique_ptr<ComputePipeline> hzbPipeline_;
    std::unique_ptr<ComputePipeline> deferredPipeline_;
    std::unique_ptr<GraphicsPipeline> streamVisibilityPipeline_;
    std::unique_ptr<ComputePipeline> streamCullResetPipeline_;
    std::unique_ptr<ComputePipeline> streamInstanceCullPipeline_;
    scene::Bounds drawBounds_;
    uint64_t sceneResourceIdentity_ = 0;
    uint64_t sceneRevision_ = 0;
    uint64_t sceneVisibilityRevision_ = 0;
    uint64_t sceneLifetimeRevision_ = 0;
    uint64_t sceneStructuralRevision_ = 0;
    uint64_t sceneContentRevision_ = 0;
    uint64_t observedHistoryInvalidationRevision_ = 0;
    uint64_t gpuSceneDrawSetRevision_ = 0;
    uint64_t gpuSceneViewAllocationId_ = 0;
    uint64_t bindingViewAllocationId_ = 0;
    uint32_t gpuSceneDrawSetGeneration_ = 0;
    GPUDrivenPreviewMeshletRange baseMeshletRange_;
    std::vector<GPUDrivenPreviewMeshletRange> lodLevelRanges_;
    std::vector<uint32_t> logicalTextureToMaterialTexture_;
    std::vector<uint32_t> streamOwnerMask_;
    std::filesystem::path compiledStreamAssetPath_;
    std::string compiledStreamSourceId_;
    std::filesystem::path compiledStreamSourcePath_;
    GPUDrivenPreviewGpuParams previousParams_;
    GPUDrivenPreviewGpuParams frozenCullingCamera_;
    uint32_t drawTaskCount_ = 0;
    uint32_t activeMeshletCount_ = 0;
    uint32_t instanceCount_ = 0;
    uint32_t materialCount_ = 1;
    uint32_t materialTextureCount_ = 1;
    uint32_t residentRecordCapacity_ = 0;
    uint32_t streamMappedInstanceCount_ = 0;
    uint32_t frameWidth_ = 1;
    uint32_t frameHeight_ = 1;
    uint32_t hzbMipCount_ = 1;
    uint64_t hzbElementCount_ = 1;
    uint32_t frameIndex_ = 0;
    uint64_t hzbHistoryEpoch_ = 0;
    uint32_t frameSlotCount_ = 0;
    uint32_t activeFrameSlot_ = 0;
    bool hzbValid_ = false;
    bool previousCameraValid_ = false;
    bool frameBuffersInitialized_ = false;
    bool cullingTargetsInitialized_ = false;
    bool freezeCullingCamera_ = false;
    bool frozenCullingCameraValid_ = false;
    bool environmentBindingValid_ = false;
    bool streamEnabled_ = false;
    uint64_t environmentResourceRevision_ = 0;
};

} // namespace

std::unique_ptr<RenderGraphPass> createGPUDrivenPreviewPass()
{
    return std::make_unique<GPUDrivenPreviewPass>();
}

} // namespace metallic::render::builtin_pass
