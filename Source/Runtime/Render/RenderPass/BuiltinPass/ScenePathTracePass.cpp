#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNrcWrapper.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include "openpbr_data_constants.h"

#if METALLIC_HAS_NRC
#include <NrcCommon.h>
#endif

#ifndef METALLIC_HAS_RTXCR
#define METALLIC_HAS_RTXCR 0
#endif

#ifndef METALLIC_RTXCR_SHADER_INCLUDE_DIR
#define METALLIC_RTXCR_SHADER_INCLUDE_DIR ""
#endif

#ifndef METALLIC_HAS_NTC
#define METALLIC_HAS_NTC 0
#endif

#ifndef METALLIC_NTC_SHADER_INCLUDE_DIR
#define METALLIC_NTC_SHADER_INCLUDE_DIR ""
#endif

namespace metallic::render::builtin_pass {
namespace {

using OpenPBRLutScalar = uint16_t;

// Radiance-cache related constants (RTXGI SHaRC / NVIDIA NRC integrations).
constexpr uint32_t kSharcDefaultEntriesLog2 = 22;
constexpr uint32_t kSharcMinEntriesLog2 = 16;
constexpr uint32_t kSharcMaxEntriesLog2 = 24;
constexpr uint32_t kSharcMaintenanceBlockSize = 256;
constexpr uint32_t kSharcDefaultMaxAccumulatedFrames = 20;
constexpr uint32_t kSharcDefaultStaleFrameNum = 60;
constexpr uint32_t kSharcDefaultUpdateStride = 5;
constexpr uint32_t kNrcMaxPathVertices = 8;

enum class PathTracePermutation : uint32_t {
    Base = 0,
    SharcUpdate,
    SharcQuery,
    NrcUpdate,
    NrcQuery,
    Count
};

constexpr const char* toString(PathTracePermutation permutation)
{
    switch (permutation) {
    case PathTracePermutation::Base:
        return "base";
    case PathTracePermutation::SharcUpdate:
        return "sharc-update";
    case PathTracePermutation::SharcQuery:
        return "sharc-query";
    case PathTracePermutation::NrcUpdate:
        return "nrc-update";
    case PathTracePermutation::NrcQuery:
        return "nrc-query";
    default:
        return "?";
    }
}

struct SceneSharcMaintenancePush {
    float cameraPosition[4] = {};
    float cameraPositionPrev[4] = {};
    float sceneScale = 1.0f;
    uint32_t entriesNum = 0;
    uint32_t accumulationFrameNum = kSharcDefaultMaxAccumulatedFrames;
    uint32_t staleFrameNumMax = kSharcDefaultStaleFrameNum;
    uint32_t frameIndex = 0;
};

// The maintenance shader reads its parameters from the front of the shared
// per-frame cache parameter buffer; keep the common prefix layout locked.
static_assert(offsetof(SceneSharcMaintenancePush, cameraPosition) ==
    offsetof(ScenePathTraceCacheParams, sharcCameraPosition));
static_assert(offsetof(SceneSharcMaintenancePush, sceneScale) ==
    offsetof(ScenePathTraceCacheParams, sharcSceneScale));
static_assert(offsetof(SceneSharcMaintenancePush, entriesNum) ==
    offsetof(ScenePathTraceCacheParams, sharcEntriesNum));
static_assert(offsetof(SceneSharcMaintenancePush, accumulationFrameNum) ==
    offsetof(ScenePathTraceCacheParams, sharcAccumulationFrameNum));
static_assert(offsetof(SceneSharcMaintenancePush, staleFrameNumMax) ==
    offsetof(ScenePathTraceCacheParams, sharcStaleFrameNumMax));
static_assert(offsetof(SceneSharcMaintenancePush, frameIndex) ==
    offsetof(ScenePathTraceCacheParams, frameIndex));

struct ScenePathTraceTonemapPush {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct OpenPBRVec3 {
    float x;
    float y;
    float z;
};

constexpr OpenPBRVec3 vec3(float x, float y, float z)
{
    return OpenPBRVec3{x, y, z};
}

static constexpr OpenPBRLutScalar kOpenPBRIdealDielectricEnergyComplement[] = {
#include "impl/data/openpbr_ideal_dielectric_energy_complement_data.h"
};

static constexpr OpenPBRLutScalar kOpenPBRIdealDielectricAverageEnergyComplement[] = {
#include "impl/data/openpbr_ideal_dielectric_avg_energy_complement_data.h"
};

static constexpr OpenPBRLutScalar kOpenPBRIdealDielectricReflectionRatio[] = {
#include "impl/data/openpbr_ideal_dielectric_reflection_ratio_data.h"
};

static constexpr OpenPBRLutScalar kOpenPBROpaqueDielectricEnergyComplement[] = {
#include "impl/data/openpbr_opaque_dielectric_energy_complement_data.h"
};

static constexpr OpenPBRLutScalar kOpenPBROpaqueDielectricAverageEnergyComplement[] = {
#include "impl/data/openpbr_opaque_dielectric_avg_energy_complement_data.h"
};

static constexpr OpenPBRLutScalar kOpenPBRIdealMetalEnergyComplement[] = {
#include "impl/data/openpbr_ideal_metal_energy_complement_data.h"
};

static constexpr OpenPBRLutScalar kOpenPBRIdealMetalAverageEnergyComplement[] = {
#include "impl/data/openpbr_ideal_metal_avg_energy_complement_data.h"
};

static constexpr OpenPBRVec3 kOpenPBRLtc[] = {
#include "impl/data/openpbr_ltc_data.h"
};

constexpr uint32_t kOpenPBRLut2DBinding = 11;
constexpr uint32_t kOpenPBRLut3DBinding = 12;
constexpr uint32_t kEnvironmentImportancePdfBinding = 13;
constexpr uint32_t kDlssRrAlbedoBinding = 14;
constexpr uint32_t kDlssRrSpecularAlbedoBinding = 15;
constexpr uint32_t kDlssRrNormalRoughnessBinding = 16;
constexpr uint32_t kDlssRrMotionVectorsBinding = 17;
constexpr uint32_t kDlssRrLinearDepthBinding = 18;
constexpr uint32_t kDlssRrSpecularHitDistanceBinding = 19;
constexpr uint32_t kOpenPBRLut2DCount = 6;
constexpr uint32_t kOpenPBRLut3DCount = 2;
constexpr uint32_t kOpenPBRLutSize = OpenPBR_EnergyTableSize;
constexpr uint32_t kOpenPBRLtcSize = OpenPBR_LTCTableSize;
constexpr float kOpenPBRLutScalarScale = 1.0f / 65535.0f;

static_assert(std::size(kOpenPBRIdealDielectricEnergyComplement) == kOpenPBRLutSize * kOpenPBRLutSize * kOpenPBRLutSize);
static_assert(std::size(kOpenPBRIdealDielectricAverageEnergyComplement) == kOpenPBRLutSize * kOpenPBRLutSize);
static_assert(std::size(kOpenPBRIdealDielectricReflectionRatio) == kOpenPBRLutSize * kOpenPBRLutSize);
static_assert(std::size(kOpenPBROpaqueDielectricEnergyComplement) == kOpenPBRLutSize * kOpenPBRLutSize * kOpenPBRLutSize);
static_assert(std::size(kOpenPBROpaqueDielectricAverageEnergyComplement) == kOpenPBRLutSize * kOpenPBRLutSize);
static_assert(std::size(kOpenPBRIdealMetalEnergyComplement) == kOpenPBRLutSize * kOpenPBRLutSize);
static_assert(std::size(kOpenPBRIdealMetalAverageEnergyComplement) == kOpenPBRLutSize);
static_assert(std::size(kOpenPBRLtc) == kOpenPBRLtcSize * kOpenPBRLtcSize);

struct OpenPBRLutTexture {
    std::unique_ptr<Buffer> uploadBuffer;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    ResourceState state = ResourceState::Undefined;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    bool uploaded = false;
};

class OpenPBRLutResources final {
public:
    Result prepare(Device& device, std::string& log)
    {
        if (valid()) {
            return {};
        }

        clear();
        Result result = createScalarLut(
            device,
            kOpenPBRIdealDielectricAverageEnergyComplement,
            kOpenPBRLutSize,
            kOpenPBRLutSize,
            1,
            "OpenPBR ideal dielectric average energy complement LUT",
            lut2D_[0],
            log);
        if (!result) {
            clear();
            return result;
        }
        result = createScalarLut(
            device,
            kOpenPBRIdealDielectricReflectionRatio,
            kOpenPBRLutSize,
            kOpenPBRLutSize,
            1,
            "OpenPBR ideal dielectric reflection ratio LUT",
            lut2D_[1],
            log);
        if (!result) {
            clear();
            return result;
        }
        result = createScalarLut(
            device,
            kOpenPBROpaqueDielectricAverageEnergyComplement,
            kOpenPBRLutSize,
            kOpenPBRLutSize,
            1,
            "OpenPBR opaque dielectric average energy complement LUT",
            lut2D_[2],
            log);
        if (!result) {
            clear();
            return result;
        }
        result = createScalarLut(
            device,
            kOpenPBRIdealMetalEnergyComplement,
            kOpenPBRLutSize,
            kOpenPBRLutSize,
            1,
            "OpenPBR ideal metal energy complement LUT",
            lut2D_[3],
            log);
        if (!result) {
            clear();
            return result;
        }
        result = createScalarLut(
            device,
            kOpenPBRIdealMetalAverageEnergyComplement,
            kOpenPBRLutSize,
            1,
            1,
            "OpenPBR ideal metal average energy complement LUT",
            lut2D_[4],
            log);
        if (!result) {
            clear();
            return result;
        }
        result = createLtcLut(device, lut2D_[5], log);
        if (!result) {
            clear();
            return result;
        }
        result = createScalarLut(
            device,
            kOpenPBRIdealDielectricEnergyComplement,
            kOpenPBRLutSize,
            kOpenPBRLutSize,
            kOpenPBRLutSize,
            "OpenPBR ideal dielectric energy complement LUT",
            lut3D_[0],
            log);
        if (!result) {
            clear();
            return result;
        }
        result = createScalarLut(
            device,
            kOpenPBROpaqueDielectricEnergyComplement,
            kOpenPBRLutSize,
            kOpenPBRLutSize,
            kOpenPBRLutSize,
            "OpenPBR opaque dielectric energy complement LUT",
            lut3D_[1],
            log);
        if (!result) {
            clear();
            return result;
        }

        refreshViews();
        return {};
    }

    Result upload(CommandBuffer& commandBuffer)
    {
        for (OpenPBRLutTexture& texture : lut2D_) {
            Result result = uploadTexture(commandBuffer, texture);
            if (!result) {
                return result;
            }
        }
        for (OpenPBRLutTexture& texture : lut3D_) {
            Result result = uploadTexture(commandBuffer, texture);
            if (!result) {
                return result;
            }
        }
        return {};
    }

    bool valid() const
    {
        return std::all_of(
            lut2D_.begin(),
            lut2D_.end(),
            [](const OpenPBRLutTexture& texture) {
                return texture.texture != nullptr && texture.view != nullptr;
            }) &&
            std::all_of(
                lut3D_.begin(),
                lut3D_.end(),
                [](const OpenPBRLutTexture& texture) {
                    return texture.texture != nullptr && texture.view != nullptr;
                });
    }

    const std::array<TextureView*, kOpenPBRLut2DCount>& lut2DViews() const
    {
        return lut2DViews_;
    }

    const std::array<TextureView*, kOpenPBRLut3DCount>& lut3DViews() const
    {
        return lut3DViews_;
    }

private:
    void clear()
    {
        for (OpenPBRLutTexture& texture : lut2D_) {
            texture = OpenPBRLutTexture{};
        }
        for (OpenPBRLutTexture& texture : lut3D_) {
            texture = OpenPBRLutTexture{};
        }
        lut2DViews_.fill(nullptr);
        lut3DViews_.fill(nullptr);
    }

    void refreshViews()
    {
        for (size_t index = 0; index < lut2D_.size(); ++index) {
            lut2DViews_[index] = lut2D_[index].view.get();
        }
        for (size_t index = 0; index < lut3D_.size(); ++index) {
            lut3DViews_[index] = lut3D_[index].view.get();
        }
    }

    template <size_t ValueCount>
    static Result createScalarLut(
        Device& device,
        const OpenPBRLutScalar (&values)[ValueCount],
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        std::string_view label,
        OpenPBRLutTexture& outTexture,
        std::string& log)
    {
        const uint64_t texelCount =
            static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * static_cast<uint64_t>(depth);
        if (texelCount != ValueCount) {
            log += "OpenPBR LUT dimensions do not match table data: ";
            log += label;
            log += '\n';
            return makeError(Error::InvalidArgument);
        }

        std::vector<float> pixels(static_cast<size_t>(texelCount) * 4u, 0.0f);
        for (size_t index = 0; index < static_cast<size_t>(texelCount); ++index) {
            pixels[index * 4u] = static_cast<float>(values[index]) * kOpenPBRLutScalarScale;
            pixels[index * 4u + 3u] = 1.0f;
        }
        return createRgbaLutTexture(device, pixels.data(), width, height, depth, label, outTexture, log);
    }

    static Result createLtcLut(Device& device, OpenPBRLutTexture& outTexture, std::string& log)
    {
        std::vector<float> pixels(std::size(kOpenPBRLtc) * 4u, 0.0f);
        for (size_t index = 0; index < std::size(kOpenPBRLtc); ++index) {
            pixels[index * 4u] = kOpenPBRLtc[index].x;
            pixels[index * 4u + 1u] = kOpenPBRLtc[index].y;
            pixels[index * 4u + 2u] = kOpenPBRLtc[index].z;
            pixels[index * 4u + 3u] = 1.0f;
        }
        return createRgbaLutTexture(
            device,
            pixels.data(),
            kOpenPBRLtcSize,
            kOpenPBRLtcSize,
            1,
            "OpenPBR LTC LUT",
            outTexture,
            log);
    }

    static Result createRgbaLutTexture(
        Device& device,
        const float* pixels,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        std::string_view label,
        OpenPBRLutTexture& outTexture,
        std::string& log)
    {
        if (pixels == nullptr || width == 0 || height == 0 || depth == 0) {
            return makeError(Error::InvalidArgument);
        }

        outTexture = OpenPBRLutTexture{};
        outTexture.width = width;
        outTexture.height = height;
        outTexture.depth = depth;
        const uint64_t byteSize =
            static_cast<uint64_t>(width) *
            static_cast<uint64_t>(height) *
            static_cast<uint64_t>(depth) *
            4ull *
            sizeof(float);
        Result result = device.createBuffer(
            BufferDesc{
                .size = byteSize,
                .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload,
            },
            outTexture.uploadBuffer);
        if (!result || outTexture.uploadBuffer == nullptr) {
            log += resultMessage(std::string("createBuffer(") + std::string(label) + " upload)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        void* mapped = outTexture.uploadBuffer->map();
        if (mapped == nullptr) {
            log += "OpenPBR LUT upload buffer map failed: ";
            log += label;
            log += '\n';
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, pixels, static_cast<size_t>(byteSize));
        outTexture.uploadBuffer->flush(0, byteSize);
        outTexture.uploadBuffer->unmap();

        result = device.createTexture(
            TextureDesc{
                .type = depth > 1 ? TextureType::Texture3D : TextureType::Texture2D,
                .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::Rgba32Sfloat,
                .width = width,
                .height = height,
                .depth = depth,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            },
            outTexture.texture);
        if (!result || outTexture.texture == nullptr) {
            log += resultMessage(std::string("createTexture(") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = device.createTextureView(
            *outTexture.texture,
            TextureViewDesc{
                .format = Format::Rgba32Sfloat,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            outTexture.view);
        if (!result || outTexture.view == nullptr) {
            log += resultMessage(std::string("createTextureView(") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    static Result uploadTexture(CommandBuffer& commandBuffer, OpenPBRLutTexture& texture)
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

    std::array<OpenPBRLutTexture, kOpenPBRLut2DCount> lut2D_;
    std::array<OpenPBRLutTexture, kOpenPBRLut3DCount> lut3D_;
    std::array<TextureView*, kOpenPBRLut2DCount> lut2DViews_{};
    std::array<TextureView*, kOpenPBRLut3DCount> lut3DViews_{};
};

struct ScenePathTraceCameraSnapshot {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
};

class ScenePathTracePass final : public ComputePass {
public:
    ~ScenePathTracePass() override = default;

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{
            EnvironmentLightingSubsystem::kSubsystemId,
        };
        return required;
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        const bool exportGuides = exportDenoiserGuides(properties());
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Path-traced glTF scene")
            .storageReadWrite()
            .format = exportGuides ? Format::Rgba16Sfloat : Format::Rgba8Unorm;
        if (exportGuides) {
            reflection.addTextureOutput("albedo", "DLSS-RR diffuse albedo guide")
                .storageReadWrite()
                .format = Format::Rgba16Sfloat;
            reflection.addTextureOutput("specularAlbedo", "DLSS-RR specular albedo guide")
                .storageReadWrite()
                .format = Format::Rgba16Sfloat;
            reflection.addTextureOutput("normalRoughness", "DLSS-RR packed normal and roughness guide")
                .storageReadWrite()
                .format = Format::Rgba16Sfloat;
            reflection.addTextureOutput("motionVectors", "DLSS-RR motion vector guide")
                .storageReadWrite()
                .format = Format::Rgba16Sfloat;
            reflection.addTextureOutput("linearDepth", "DLSS-RR linear depth guide")
                .storageReadWrite()
                .format = Format::R32Sfloat;
            reflection.addTextureOutput("specularHitDistance", "DLSS-RR specular hit distance guide")
                .storageReadWrite()
                .format = Format::R32Sfloat;
        }
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeIntSetting(
                "maxDepth",
                "Max Depth",
                static_cast<int32_t>(kDefaultPathTraceMaxDepth),
                1,
                static_cast<int32_t>(kMaxPathTraceMaxDepth),
                true),
            runtimeIntSetting(
                "samples",
                "Samples",
                static_cast<int32_t>(kDefaultPathTraceSamples),
                1,
                static_cast<int32_t>(kMaxPathTraceSamples),
                true),
            runtimeBoolSetting("accumulate", "Accumulate", true, true),
            runtimeBoolSetting("flipBitangent", "Flip Bitangent", false, true),
            runtimeEnumSetting(
                "cacheMode",
                "Radiance Cache",
                "off",
                {
                    {"Off", "off"},
                    {"SHaRC (RTXGI)", "sharc"},
                    {"NRC (NVIDIA)", "nrc"},
                },
                true),
            runtimeIntSetting(
                "sharc.entriesLog2",
                "SHaRC Entries (log2)",
                static_cast<int32_t>(kSharcDefaultEntriesLog2),
                static_cast<int32_t>(kSharcMinEntriesLog2),
                static_cast<int32_t>(kSharcMaxEntriesLog2),
                true),
            runtimeFloatSetting("sharc.sceneScale", "SHaRC Scene Scale", 0.0f, 0.0f, 1000.0f, true),
            runtimeIntSetting(
                "sharc.maxAccumulatedFrames",
                "SHaRC Max Accumulated Frames",
                static_cast<int32_t>(kSharcDefaultMaxAccumulatedFrames),
                1,
                1024,
                false),
            runtimeIntSetting(
                "sharc.staleFrameNum",
                "SHaRC Stale Frame Num",
                static_cast<int32_t>(kSharcDefaultStaleFrameNum),
                8,
                1024,
                false),
            runtimeIntSetting(
                "sharc.updateStride",
                "SHaRC Update Stride",
                static_cast<int32_t>(kSharcDefaultUpdateStride),
                1,
                16,
                false),
#if METALLIC_HAS_NRC
            runtimeFloatSetting("nrc.maxExpectedRadiance", "NRC Max Expected Radiance", 1.0f, 0.01f, 100.0f, false),
            runtimeEnumSetting(
                "nrc.resolveMode",
                "NRC Resolve Mode",
                "add",
                {
                    {"Add Query Result", "add"},
                    {"Replace Output", "replace"},
                    {"Training Bounce Heatmap", "heatmap"},
                    {"Query Index", "queryIndex"},
                    {"Direct Cache View", "cacheView"},
                },
                false),
#endif
        };
        appendCameraRuntimeSettings(
            settings,
            std::array<float, 3>{0.0f, 0.2f, 2.5f},
            std::array<float, 3>{0.0f, 0.0f, 0.0f},
            50.0f,
            true);
        return settings;
    }
    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "ScenePathTracePass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().rayTracingAccelerationStructure ||
            !context.device->capabilities().rayQuery) {
            log = "ScenePathTracePass requires rayTracingAccelerationStructure and rayQuery capabilities";
            return makeError(Error::Unsupported);
        }
        device_ = context.device;
        graphicsQueue_ = context.graphicsQueue;
        sceneResourceManager_ = context.sceneResourceManager;

        Result result;
        if (context.sceneResourceManager != nullptr) {
            std::shared_ptr<SceneResourceSnapshot> snapshot;
            result = context.sceneResourceManager->acquire(
                *context.device,
                *context.graphicsQueue,
                properties(),
                context.runtimeScene,
                SceneResourceFeatureBits::Geometry |
                    SceneResourceFeatureBits::Materials |
                    SceneResourceFeatureBits::MaterialTextures |
                    SceneResourceFeatureBits::StandardAccelerationStructure,
                snapshot,
                log);
            if (result && snapshot != nullptr) {
                sceneResources_ = *snapshot->pathTraceResources;
            }
        } else {
            result = sceneResources_.prepare(
                *context.device,
                *context.graphicsQueue,
                properties(),
                context.runtimeScene,
                log);
        }
        if (!result) {
            return result;
        }
        const uint64_t resourceRevision = sceneResources_.revision();
        if (resourceRevision != sceneResourceRevision_) {
            sceneResourceRevision_ = resourceRevision;
            resetAccumulation_ = true;
            hasPreviousCamera_ = false;
        }
        const std::string bsdf = stringProperty(properties(), "bsdf", "standard");
        const bool useOpenPBR = bsdf == "openpbr" || bsdf == "OpenPBR";
        const bool exportGuides = exportDenoiserGuides(properties());
        const bool ntcActive = sceneResources_.neuralTextures().active();
        const bool ntcCooperativeVector =
            sceneResources_.neuralTextures().cooperativeVectorActive();
        const char* moduleName = nullptr;
        const char* entryPointName = nullptr;
        if (useOpenPBR) {
            moduleName = exportGuides
                ? kOpenPBRRayQueryPathTraceGuidesShaderModuleName
                : kOpenPBRRayQueryPathTraceShaderModuleName;
            entryPointName = exportGuides
                ? kOpenPBRRayQueryPathTraceGuidesEntryPoint
                : kOpenPBRRayQueryPathTraceEntryPoint;
        } else {
            moduleName = exportGuides
                ? kScenePathTraceGuidesShaderModuleName
                : kScenePathTraceShaderModuleName;
            entryPointName = exportGuides
                ? kScenePathTraceGuidesEntryPoint
                : kScenePathTraceEntryPoint;
        }
        const uint32_t requestedCacheMode = cacheModeFromProperties(properties());
        uint32_t cacheMode = requestedCacheMode;
        std::string cacheWarning;
        if (cacheMode != kScenePathTraceCacheModeOff) {
            if (useOpenPBR || exportGuides) {
                cacheMode = kScenePathTraceCacheModeOff;
                cacheWarning =
                    "ScenePathTracePass radiance cache requires the standard BSDF without denoiser guides; cache disabled\n";
            }
#if METALLIC_HAS_NRC
            else if (cacheMode == kScenePathTraceCacheModeNrc && !context.device->capabilities().rayQuery) {
                cacheMode = kScenePathTraceCacheModeOff;
            }
#else
            else if (cacheMode == kScenePathTraceCacheModeNrc) {
                cacheMode = kScenePathTraceCacheModeOff;
                cacheWarning =
                    "ScenePathTracePass built without the NRC SDK (METALLIC_HAS_NRC=0); NRC cache disabled\n";
            }
#endif
        }
        cacheMode_ = cacheMode;
        if (!cacheWarning.empty()) {
            log += cacheWarning;
        }

        const std::string shaderKey = std::string(moduleName) + "." + entryPointName +
            "|cache=" + std::to_string(cacheMode_) +
            "|ntc=" + (ntcActive ? "1" : "0") +
            "|coopvec=" + (ntcCooperativeVector ? "1" : "0");
        if (useOpenPBR) {
            result = openPBRLuts_.prepare(*context.device, log);
            if (!result) {
                clearPrograms();
                compiledShaderKey_.clear();
                return result;
            }
        }
        if (compiledShaderKey_ != shaderKey) {
            clearPrograms();
            compiledShaderKey_.clear();
            resetAccumulation_ = true;
            hasPreviousCamera_ = false;
            sharcResourcesRevision_ = 0;
        }

        const bool baseReady = programs_[static_cast<size_t>(PathTracePermutation::Base)].valid();
        const bool sharcReady = cacheMode_ != kScenePathTraceCacheModeSharc ||
            (programs_[static_cast<size_t>(PathTracePermutation::SharcUpdate)].valid() &&
                programs_[static_cast<size_t>(PathTracePermutation::SharcQuery)].valid() &&
                sharcClearProgram_.valid() && sharcResolveProgram_.valid());
        const bool nrcReady = cacheMode_ != kScenePathTraceCacheModeNrc ||
            (programs_[static_cast<size_t>(PathTracePermutation::NrcUpdate)].valid() &&
                programs_[static_cast<size_t>(PathTracePermutation::NrcQuery)].valid() &&
                tonemapProgram_.valid());
        if (baseReady && sharcReady && nrcReady) {
            return {};
        }

        std::vector<const char*> capabilities{
            "spvRayQueryKHR",
            "spvGroupNonUniformBallot",
        };
        if (ntcCooperativeVector) {
            capabilities.push_back("spvCooperativeVectorNV");
        }

        // Keep the conventional binding table stable; append NTC descriptors only when active.
        std::vector<ComputeProgramBindingDesc> baseBindings{
            ComputeProgramBindingDesc{
                .binding = 0,
                .kind = ComputeResourceBindingKind::AccelerationStructure,
            },
            ComputeProgramBindingDesc{
                .binding = 1,
                .kind = ComputeResourceBindingKind::StorageImage,
            },
            ComputeProgramBindingDesc{
                .binding = 2,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeProgramBindingDesc{
                .binding = 3,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeProgramBindingDesc{
                .binding = 4,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeProgramBindingDesc{
                .binding = 5,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeProgramBindingDesc{
                .binding = 6,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeProgramBindingDesc{
                .binding = 7,
                .kind = ComputeResourceBindingKind::StorageImage,
            },
            ComputeProgramBindingDesc{
                .binding = 8,
                .kind = ComputeResourceBindingKind::StorageImage,
            },
            ComputeProgramBindingDesc{
                .binding = 9,
                .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = kScenePathTraceMaxMaterialTextures,
            },
            ComputeProgramBindingDesc{
                .binding = 10,
                .kind = ComputeResourceBindingKind::SampledImage,
            },
            ComputeProgramBindingDesc{
                .binding = kEnvironmentImportancePdfBinding,
                .kind = ComputeResourceBindingKind::SampledImage,
            },
        };
        if (useOpenPBR) {
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kOpenPBRLut2DBinding,
                .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = kOpenPBRLut2DCount,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kOpenPBRLut3DBinding,
                .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = kOpenPBRLut3DCount,
            });
        }
        if (exportGuides) {
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kDlssRrAlbedoBinding,
                .kind = ComputeResourceBindingKind::StorageImage,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kDlssRrSpecularAlbedoBinding,
                .kind = ComputeResourceBindingKind::StorageImage,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kDlssRrNormalRoughnessBinding,
                .kind = ComputeResourceBindingKind::StorageImage,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kDlssRrMotionVectorsBinding,
                .kind = ComputeResourceBindingKind::StorageImage,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kDlssRrLinearDepthBinding,
                .kind = ComputeResourceBindingKind::StorageImage,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kDlssRrSpecularHitDistanceBinding,
                .kind = ComputeResourceBindingKind::StorageImage,
            });
        }
        if (ntcActive) {
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kNeuralTextureLatentsBinding,
                .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = kMaxNeuralTextureSets,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kNeuralTextureConstantsBinding,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kNeuralTextureWeightsBinding,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kNeuralTextureSetInfoBinding,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            baseBindings.push_back(ComputeProgramBindingDesc{
                .binding = kNeuralTextureSamplerBinding,
                .kind = ComputeResourceBindingKind::Sampler,
            });
        }

        auto compilePermutation =
            [&](PathTracePermutation permutation,
                std::span<const SlangMacroDefine> extraDefines,
                const std::vector<ComputeProgramBindingDesc>& permutationBindings,
                ComputeProgram& outProgram) -> Result {
            std::vector<SlangMacroDefine> defines{
                SlangMacroDefine{
                    .name = "METALLIC_HAS_RTXCR",
                    .value = METALLIC_HAS_RTXCR ? "1" : "0",
                },
                SlangMacroDefine{
                    .name = "METALLIC_HAS_NTC",
                    .value = ntcActive ? "1" : "0",
                },
                SlangMacroDefine{
                    .name = "METALLIC_NTC_COOPERATIVE_VECTOR",
                    .value = ntcCooperativeVector ? "1" : "0",
                },
            };
            defines.insert(defines.end(), extraDefines.begin(), extraDefines.end());
            std::vector<const char*> additionalSearchPaths;
#if METALLIC_HAS_RTXCR
            additionalSearchPaths.push_back(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
#endif
#if METALLIC_HAS_NTC
            if (ntcActive) {
                additionalSearchPaths.push_back(METALLIC_NTC_SHADER_INCLUDE_DIR);
            }
#endif
            ShaderCompileResult permutationCompile;
            Result permutationResult = compileSlangShaderToSpirv(
                SlangShaderDesc{
                    .moduleName = moduleName,
                    .entryPointName = entryPointName,
                    .searchPath = kTriangleShaderSearchPath,
                    .additionalSearchPaths = additionalSearchPaths.data(),
                    .additionalSearchPathCount =
                        static_cast<uint32_t>(additionalSearchPaths.size()),
                    .capabilities = capabilities.data(),
                    .capabilityCount = static_cast<uint32_t>(capabilities.size()),
                    .macroDefines = defines.data(),
                    .macroDefineCount = static_cast<uint32_t>(defines.size()),
                },
                permutationCompile);
            if (!permutationResult) {
                log += "compileSlangShaderToSpirv(";
                log += moduleName;
                log += ".";
                log += entryPointName;
                log += "[";
                log += toString(permutation);
                log += "]) returned ";
                log += resultToString(permutationResult);
                if (!permutationCompile.diagnostics.empty()) {
                    log += ": ";
                    log += permutationCompile.diagnostics;
                }
                log += '\n';
                outProgram.clear();
                return permutationResult;
            }

            std::string programLog;
            const std::string debugName = std::string("ScenePathTracePass.") + toString(permutation);
            permutationResult = outProgram.initialize(
                *context.device,
                ComputeProgramDesc{
                    .spirv = permutationCompile.spirv.data(),
                    .byteSize = static_cast<uint64_t>(permutationCompile.spirv.size() * sizeof(uint32_t)),
                    .pushConstantSize = sizeof(ScenePathTracePush),
                    .bindings = permutationBindings.data(),
                    .bindingCount = static_cast<uint32_t>(permutationBindings.size()),
                    .debugName = debugName.c_str(),
                },
                programLog);
            if (!programLog.empty()) {
                if (!log.empty() && log.back() != '\n') {
                    log += '\n';
                }
                log += programLog;
            }
            if (!permutationResult) {
                outProgram.clear();
                return permutationResult;
            }
            return {};
        };

        const std::vector<ComputeProgramBindingDesc> cacheBindings = [baseBindings]() {
            std::vector<ComputeProgramBindingDesc> bindings = baseBindings;
            bindings.push_back(ComputeProgramBindingDesc{
                .binding = kScenePathTraceCacheParamsBinding,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            return bindings;
        }();

        if (!programs_[static_cast<size_t>(PathTracePermutation::Base)].valid()) {
            result = compilePermutation(
                PathTracePermutation::Base,
                {},
                baseBindings,
                programs_[static_cast<size_t>(PathTracePermutation::Base)]);
            if (!result) {
                return result;
            }
        }

        if (cacheMode_ == kScenePathTraceCacheModeSharc) {
            const std::vector<ComputeProgramBindingDesc> sharcBindings = [cacheBindings]() {
                std::vector<ComputeProgramBindingDesc> bindings = cacheBindings;
                bindings.push_back(ComputeProgramBindingDesc{
                    .binding = kScenePathTraceSharcHashEntriesBinding,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
                });
                bindings.push_back(ComputeProgramBindingDesc{
                    .binding = kScenePathTraceSharcAccumulationBinding,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
                });
                bindings.push_back(ComputeProgramBindingDesc{
                    .binding = kScenePathTraceSharcResolvedBinding,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
                });
                return bindings;
            }();

            const std::array<SlangMacroDefine, 1> sharcUpdateDefines{
                SlangMacroDefine{.name = "SHARC_UPDATE", .value = "1"},
            };
            if (!programs_[static_cast<size_t>(PathTracePermutation::SharcUpdate)].valid()) {
                result = compilePermutation(
                    PathTracePermutation::SharcUpdate,
                    sharcUpdateDefines,
                    sharcBindings,
                    programs_[static_cast<size_t>(PathTracePermutation::SharcUpdate)]);
                if (!result) {
                    return result;
                }
            }

            const std::array<SlangMacroDefine, 1> sharcQueryDefines{
                SlangMacroDefine{.name = "SHARC_QUERY", .value = "1"},
            };
            if (!programs_[static_cast<size_t>(PathTracePermutation::SharcQuery)].valid()) {
                result = compilePermutation(
                    PathTracePermutation::SharcQuery,
                    sharcQueryDefines,
                    sharcBindings,
                    programs_[static_cast<size_t>(PathTracePermutation::SharcQuery)]);
                if (!result) {
                    return result;
                }
            }

            // SHaRC maintenance programs (clear + resolve).
            const std::array<ComputeProgramBindingDesc, 4> maintenanceBindings{
                ComputeProgramBindingDesc{
                    .binding = 0,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
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
            auto compileMaintenance =
                [&](const char* entryPointName, ComputeProgram& outProgram) -> Result {
                ShaderCompileResult maintenanceCompile;
                Result maintenanceResult = compileSlangShaderToSpirv(
                    SlangShaderDesc{
                        .moduleName = kSceneSharcMaintenanceShaderModuleName,
                        .entryPointName = entryPointName,
                        .searchPath = kTriangleShaderSearchPath,
                        .capabilities = capabilities.data(),
                        .capabilityCount = static_cast<uint32_t>(capabilities.size()),
                    },
                    maintenanceCompile);
                if (!maintenanceResult) {
                    log += "compileSlangShaderToSpirv(";
                    log += kSceneSharcMaintenanceShaderModuleName;
                    log += ".";
                    log += entryPointName;
                    log += ") returned ";
                    log += resultToString(maintenanceResult);
                    if (!maintenanceCompile.diagnostics.empty()) {
                        log += ": ";
                        log += maintenanceCompile.diagnostics;
                    }
                    log += '\n';
                    outProgram.clear();
                    return maintenanceResult;
                }
                std::string programLog;
                const std::string maintenanceDebugName =
                    std::string("ScenePathTracePass.") + entryPointName;
                maintenanceResult = outProgram.initialize(
                    *context.device,
                    ComputeProgramDesc{
                        .spirv = maintenanceCompile.spirv.data(),
                        .byteSize = static_cast<uint64_t>(maintenanceCompile.spirv.size() * sizeof(uint32_t)),
                        .pushConstantSize = sizeof(SceneSharcMaintenancePush),
                        .bindings = maintenanceBindings.data(),
                        .bindingCount = static_cast<uint32_t>(maintenanceBindings.size()),
                        .debugName = maintenanceDebugName.c_str(),
                        .requiresRayQuery = false,
                    },
                    programLog);
                if (!programLog.empty()) {
                    if (!log.empty() && log.back() != '\n') {
                        log += '\n';
                    }
                    log += programLog;
                }
                if (!maintenanceResult) {
                    outProgram.clear();
                }
                return maintenanceResult;
            };
            if (!sharcClearProgram_.valid()) {
                result = compileMaintenance("sharcClearMain", sharcClearProgram_);
                if (!result) {
                    return result;
                }
            }
            if (!sharcResolveProgram_.valid()) {
                result = compileMaintenance("sharcResolveMain", sharcResolveProgram_);
                if (!result) {
                    return result;
                }
            }
        }

#if METALLIC_HAS_NRC
        if (cacheMode_ == kScenePathTraceCacheModeNrc) {
            const std::vector<ComputeProgramBindingDesc> nrcBindings = [cacheBindings]() {
                std::vector<ComputeProgramBindingDesc> bindings = cacheBindings;
                bindings.push_back(ComputeProgramBindingDesc{
                    .binding = kScenePathTraceNrcQueryPathInfoBinding,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
                });
                bindings.push_back(ComputeProgramBindingDesc{
                    .binding = kScenePathTraceNrcTrainingPathInfoBinding,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
                });
                bindings.push_back(ComputeProgramBindingDesc{
                    .binding = kScenePathTraceNrcTrainingPathVerticesBinding,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
                });
                bindings.push_back(ComputeProgramBindingDesc{
                    .binding = kScenePathTraceNrcQueryRadianceParamsBinding,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
                });
                bindings.push_back(ComputeProgramBindingDesc{
                    .binding = kScenePathTraceNrcCountersBinding,
                    .kind = ComputeResourceBindingKind::StorageBuffer,
                });
                return bindings;
            }();

            const std::array<SlangMacroDefine, 1> nrcUpdateDefines{
                SlangMacroDefine{.name = "NRC_UPDATE", .value = "1"},
            };
            if (!programs_[static_cast<size_t>(PathTracePermutation::NrcUpdate)].valid()) {
                result = compilePermutation(
                    PathTracePermutation::NrcUpdate,
                    nrcUpdateDefines,
                    nrcBindings,
                    programs_[static_cast<size_t>(PathTracePermutation::NrcUpdate)]);
                if (!result) {
                    return result;
                }
            }

            const std::array<SlangMacroDefine, 1> nrcQueryDefines{
                SlangMacroDefine{.name = "NRC_QUERY", .value = "1"},
            };
            if (!programs_[static_cast<size_t>(PathTracePermutation::NrcQuery)].valid()) {
                result = compilePermutation(
                    PathTracePermutation::NrcQuery,
                    nrcQueryDefines,
                    nrcBindings,
                    programs_[static_cast<size_t>(PathTracePermutation::NrcQuery)]);
                if (!result) {
                    return result;
                }
            }

            // Tonemap pass producing the final displayable color after the
            // NRC resolve has added the predicted radiance.
            if (!tonemapProgram_.valid()) {
                const std::array<ComputeProgramBindingDesc, 2> tonemapBindings{
                    ComputeProgramBindingDesc{
                        .binding = 0,
                        .kind = ComputeResourceBindingKind::StorageImage,
                    },
                    ComputeProgramBindingDesc{
                        .binding = 1,
                        .kind = ComputeResourceBindingKind::StorageImage,
                    },
                };
                ShaderCompileResult tonemapCompile;
                Result tonemapResult = compileSlangShaderToSpirv(
                    SlangShaderDesc{
                        .moduleName = kScenePathTraceTonemapShaderModuleName,
                        .entryPointName = kScenePathTraceTonemapEntryPointName,
                        .searchPath = kTriangleShaderSearchPath,
                        .capabilities = capabilities.data(),
                        .capabilityCount = static_cast<uint32_t>(capabilities.size()),
                    },
                    tonemapCompile);
                if (!tonemapResult) {
                    log += "compileSlangShaderToSpirv(";
                    log += kScenePathTraceTonemapShaderModuleName;
                    log += ") returned ";
                    log += resultToString(tonemapResult);
                    if (!tonemapCompile.diagnostics.empty()) {
                        log += ": ";
                        log += tonemapCompile.diagnostics;
                    }
                    log += '\n';
                    tonemapProgram_.clear();
                    return tonemapResult;
                }
                std::string programLog;
                tonemapResult = tonemapProgram_.initialize(
                    *context.device,
                    ComputeProgramDesc{
                        .spirv = tonemapCompile.spirv.data(),
                        .byteSize = static_cast<uint64_t>(tonemapCompile.spirv.size() * sizeof(uint32_t)),
                        .pushConstantSize = sizeof(ScenePathTraceTonemapPush),
                        .bindings = tonemapBindings.data(),
                        .bindingCount = static_cast<uint32_t>(tonemapBindings.size()),
                        .debugName = "ScenePathTracePass.Tonemap",
                        .requiresRayQuery = false,
                    },
                    programLog);
                if (!programLog.empty()) {
                    if (!log.empty() && log.back() != '\n') {
                        log += '\n';
                    }
                    log += programLog;
                }
                if (!tonemapResult) {
                    tonemapProgram_.clear();
                    return tonemapResult;
                }
            }
        }
#else
        if (cacheMode_ == kScenePathTraceCacheModeNrc) {
            // Already downgraded to off above; nothing to compile.
        }
#endif

        compiledShaderKey_ = shaderKey;
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        std::string syncLog;
        if (sceneResourceManager_ != nullptr && device_ != nullptr && graphicsQueue_ != nullptr) {
            std::shared_ptr<SceneResourceSnapshot> snapshot;
            Result acquireResult = sceneResourceManager_->acquire(
                *device_,
                *graphicsQueue_,
                context.properties(),
                context.runtimeScene(),
                SceneResourceFeatureBits::Geometry |
                    SceneResourceFeatureBits::Materials |
                    SceneResourceFeatureBits::MaterialTextures |
                    SceneResourceFeatureBits::StandardAccelerationStructure,
                snapshot,
                syncLog);
            if (!acquireResult || snapshot == nullptr) {
                return acquireResult ? makeError(Error::Failure) : acquireResult;
            }
            sceneResources_ = *snapshot->pathTraceResources;
        }
        Result syncResult = sceneResources_.syncRuntimeScene(context.runtimeScene(), syncLog);
        if (!syncResult) {
            spdlog::warn("[ScenePathTracePass] Runtime scene sync failed: {}", syncLog);
            return syncResult;
        }
        if (!sceneResources_.textureUploadsReady()) {
            return {};
        }
        if (sceneResources_.revision() != sceneResourceRevision_) {
            sceneResourceRevision_ = sceneResources_.revision();
            resetAccumulation_ = true;
            hasPreviousCamera_ = false;
        }
        EnvironmentLightingSubsystem* environmentSubsystem =
            context.subsystem<EnvironmentLightingSubsystem>();
        if (environmentSubsystem == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        const EnvironmentLightingSnapshot& environment = environmentSubsystem->snapshot();
        if (!environment.valid()) {
            return {};
        }
        if (environment.resourceRevision != environmentResourceRevision_ ||
            environment.settingsRevision != environmentSettingsRevision_) {
            environmentResourceRevision_ = environment.resourceRevision;
            environmentSettingsRevision_ = environment.settingsRevision;
            resetAccumulation_ = true;
            hasPreviousCamera_ = false;
        }
        TextureHandle color = context.outputTexture("color");
        const auto& materialTextureViews = sceneResources_.materialTextureViews();
        TextureView* environmentTextureView = environment.radianceView;
        TextureView* environmentImportancePdfView = environment.pdfView;
        TextureView* const environmentTextureViews[] = {environmentTextureView};
        TextureView* const environmentImportancePdfViews[] = {environmentImportancePdfView};
        const bool useOpenPBR = useOpenPBRBsdf(properties());
        const bool exportGuides = exportDenoiserGuides(properties());
        TextureHandle albedo = exportGuides ? context.outputTexture("albedo") : TextureHandle{};
        TextureHandle specularAlbedo = exportGuides ? context.outputTexture("specularAlbedo") : TextureHandle{};
        TextureHandle normalRoughness = exportGuides ? context.outputTexture("normalRoughness") : TextureHandle{};
        TextureHandle motionVectors = exportGuides ? context.outputTexture("motionVectors") : TextureHandle{};
        TextureHandle linearDepth = exportGuides ? context.outputTexture("linearDepth") : TextureHandle{};
        TextureHandle specularHitDistance = exportGuides ? context.outputTexture("specularHitDistance") : TextureHandle{};

        uint32_t cacheMode = cacheMode_;
        ComputeProgram* renderProgram = &programs_[static_cast<size_t>(PathTracePermutation::Base)];
        if (cacheMode == kScenePathTraceCacheModeSharc) {
            if (!programs_[static_cast<size_t>(PathTracePermutation::SharcQuery)].valid() ||
                !programs_[static_cast<size_t>(PathTracePermutation::SharcUpdate)].valid() ||
                !sharcClearProgram_.valid() ||
                !sharcResolveProgram_.valid()) {
                cacheMode = kScenePathTraceCacheModeOff;
                renderProgram = &programs_[static_cast<size_t>(PathTracePermutation::Base)];
            } else {
                renderProgram = &programs_[static_cast<size_t>(PathTracePermutation::SharcQuery)];
            }
        } else if (cacheMode == kScenePathTraceCacheModeNrc) {
#if METALLIC_HAS_NRC
            if (!programs_[static_cast<size_t>(PathTracePermutation::NrcQuery)].valid() ||
                !programs_[static_cast<size_t>(PathTracePermutation::NrcUpdate)].valid() ||
                !tonemapProgram_.valid()) {
                cacheMode = kScenePathTraceCacheModeOff;
                renderProgram = &programs_[static_cast<size_t>(PathTracePermutation::Base)];
            }
#else
            cacheMode = kScenePathTraceCacheModeOff;
            renderProgram = &programs_[static_cast<size_t>(PathTracePermutation::Base)];
#endif
        }
        cacheMode_ = cacheMode;

        if (!color.valid() ||
            color.view() == nullptr ||
            renderProgram == nullptr ||
            !renderProgram->valid() ||
            !sceneResources_.valid() ||
            materialTextureViews[0] == nullptr ||
            environmentTextureView == nullptr ||
            environmentImportancePdfView == nullptr ||
            (useOpenPBR && !openPBRLuts_.valid()) ||
            (exportGuides &&
                (!validTexture(albedo) ||
                    !validTexture(specularAlbedo) ||
                    !validTexture(normalRoughness) ||
                    !validTexture(motionVectors) ||
                    !validTexture(linearDepth) ||
                    !validTexture(specularHitDistance)))) {
            return makeError(Error::InvalidArgument);
        }

        ScenePathTracePush push;
        buildPush(
            context.width(),
            context.height(),
            context.properties(),
            sceneResources_.bounds(),
            environment.settings,
            environment.mapAvailable,
            push);
        push.materialTextureCount = sceneResources_.materialTextureCount();
        push.ntcTextureSetCount = sceneResources_.neuralTextures().textureSetCount();
        push.cacheMode = cacheMode;
        push.outputLinear = cacheMode == kScenePathTraceCacheModeNrc ? 1u : 0u;
        const ScenePathTraceCameraSnapshot currentCamera = cameraSnapshotFromPush(push);
        const bool previousCameraValid =
            hasPreviousCamera_ &&
            previousCameraWidth_ == context.width() &&
            previousCameraHeight_ == context.height();
        applyPreviousCameraSnapshot(previousCameraValid ? previousCamera_ : currentCamera, push);
        push.previousCameraValid = previousCameraValid ? 1u : 0u;

        TextureView* historyCurrentView = color.view();
        TextureView* historyPreviousView = color.view();
        Result result = prepareHistoryTextures(
            context,
            *color.view(),
            push,
            historyCurrentView,
            historyPreviousView);
        if (!result) {
            return result;
        }
        if (historyCurrentView == nullptr || historyPreviousView == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        result = sceneResources_.uploadMaterialTextures(context.commandBuffer());
        if (!result) {
            return result;
        }
        if (useOpenPBR) {
            result = openPBRLuts_.upload(context.commandBuffer());
            if (!result) {
                return result;
            }
        }

        std::vector<ComputeDispatchBinding> bindings{
            ComputeDispatchBinding{
                .binding = 0,
                .accelerationStructure =
                    sceneResources_.accelerationStructure().accelerationStructure(),
            },
            ComputeDispatchBinding{
                .binding = 1,
                .textureView = color.view(),
            },
            ComputeDispatchBinding{
                .binding = 2,
                .buffer = sceneResources_.vertexBuffer(),
            },
            ComputeDispatchBinding{
                .binding = 3,
                .buffer = sceneResources_.indexBuffer(),
            },
            ComputeDispatchBinding{
                .binding = 4,
                .buffer = sceneResources_.primitiveBuffer(),
            },
            ComputeDispatchBinding{
                .binding = 5,
                .buffer = sceneResources_.instanceBuffer(),
            },
            ComputeDispatchBinding{
                .binding = 6,
                .buffer = sceneResources_.materialBuffer(),
            },
            ComputeDispatchBinding{
                .binding = 7,
                .textureView = historyCurrentView,
            },
            ComputeDispatchBinding{
                .binding = 8,
                .textureView = historyPreviousView,
            },
            ComputeDispatchBinding{
                .binding = 9,
                .textureViews = materialTextureViews.data(),
                .textureViewCount = static_cast<uint32_t>(materialTextureViews.size()),
            },
            ComputeDispatchBinding{
                .binding = 10,
                .textureViews = environmentTextureViews,
                .textureViewCount = static_cast<uint32_t>(std::size(environmentTextureViews)),
            },
            ComputeDispatchBinding{
                .binding = kEnvironmentImportancePdfBinding,
                .textureViews = environmentImportancePdfViews,
                .textureViewCount = static_cast<uint32_t>(std::size(environmentImportancePdfViews)),
            },
        };
        if (useOpenPBR) {
            const auto& lut2DViews = openPBRLuts_.lut2DViews();
            const auto& lut3DViews = openPBRLuts_.lut3DViews();
            bindings.push_back(ComputeDispatchBinding{
                .binding = kOpenPBRLut2DBinding,
                .textureViews = lut2DViews.data(),
                .textureViewCount = static_cast<uint32_t>(lut2DViews.size()),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kOpenPBRLut3DBinding,
                .textureViews = lut3DViews.data(),
                .textureViewCount = static_cast<uint32_t>(lut3DViews.size()),
            });
        }
        if (exportGuides) {
            bindings.push_back(ComputeDispatchBinding{
                .binding = kDlssRrAlbedoBinding,
                .textureView = albedo.view(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kDlssRrSpecularAlbedoBinding,
                .textureView = specularAlbedo.view(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kDlssRrNormalRoughnessBinding,
                .textureView = normalRoughness.view(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kDlssRrMotionVectorsBinding,
                .textureView = motionVectors.view(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kDlssRrLinearDepthBinding,
                .textureView = linearDepth.view(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kDlssRrSpecularHitDistanceBinding,
                .textureView = specularHitDistance.view(),
            });
        }
        const NeuralTextureResources& neuralTextures = sceneResources_.neuralTextures();
        if (neuralTextures.active()) {
            const auto& latentViews = neuralTextures.latentTextureViews();
            bindings.push_back(ComputeDispatchBinding{
                .binding = kNeuralTextureLatentsBinding,
                .textureViews = latentViews.data(),
                .textureViewCount = static_cast<uint32_t>(latentViews.size()),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kNeuralTextureConstantsBinding,
                .buffer = neuralTextures.constantsBuffer(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kNeuralTextureWeightsBinding,
                .buffer = neuralTextures.weightsBuffer(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kNeuralTextureSetInfoBinding,
                .buffer = neuralTextures.setInfoBuffer(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = kNeuralTextureSamplerBinding,
                .sampler = &neuralTextures.latentSampler(),
            });
        }

        if (cacheMode != kScenePathTraceCacheModeOff) {
            result = ensureCacheParamsBuffer(*device_);
            if (!result) {
                return result;
            }
        }

        if (cacheMode == kScenePathTraceCacheModeSharc) {
            result = executeSharcFrame(
                context,
                push,
                bindings,
                *renderProgram,
                programs_[static_cast<size_t>(PathTracePermutation::SharcUpdate)]);
            if (!result) {
                return result;
            }
        } else if (cacheMode == kScenePathTraceCacheModeNrc) {
#if METALLIC_HAS_NRC
            result = executeNrcFrame(
                context,
                push,
                bindings,
                programs_[static_cast<size_t>(PathTracePermutation::NrcUpdate)],
                programs_[static_cast<size_t>(PathTracePermutation::NrcQuery)],
                historyCurrentView);
            if (!result) {
                return result;
            }
#endif
        } else {
            result = renderProgram->dispatch(ComputeDispatchDesc{
                .commandBuffer = &context.commandBuffer(),
                .bindings = bindings.data(),
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pushData = &push,
                .pushDataSize = sizeof(push),
                .groupCountX = (context.width() + 7) / 8,
                .groupCountY = (context.height() + 7) / 8,
                .groupCountZ = 1,
            });
            if (!result) {
                return result;
            }
        }

        if (push.enableAccumulation != 0 && context.historyResources() != nullptr) {
            context.historyResources()->markWritten(historyNameForContext(context));
        }
        previousCamera_ = currentCamera;
        previousCameraWidth_ = context.width();
        previousCameraHeight_ = context.height();
        hasPreviousCamera_ = true;
        return {};
    }

private:
    static bool validTexture(TextureHandle texture)
    {
        return texture.valid() && texture.texture() != nullptr && texture.view() != nullptr;
    }

    static bool exportDenoiserGuides(const RenderGraphProperties& properties)
    {
        return boolProperty(properties, "exportDenoiserGuides", false);
    }

    Result prepareHistoryTextures(
        RenderGraphExecutionContext& context,
        TextureView& fallbackView,
        ScenePathTracePush& push,
        TextureView*& outCurrentView,
        TextureView*& outPreviousView)
    {
        HistoryResourceManager* history = context.historyResources();
        // NRC mode always routes through the linear HDR history texture: the
        // resolve pass adds predicted radiance before a separate tonemap pass.
        const bool accumulationEnabled = push.cacheMode == kScenePathTraceCacheModeNrc ||
            boolProperty(context.properties(), "accumulate", true);
        push.enableAccumulation = accumulationEnabled && history != nullptr ? 1u : 0u;
        push.hasHistory = 0;
        push.accumulationFrame = 0;
        outCurrentView = &fallbackView;
        outPreviousView = &fallbackView;
        if (push.enableAccumulation == 0) {
            accumulationFrame_ = 0;
            resetAccumulation_ = true;
            return {};
        }

        const bool nrcHistory = push.cacheMode == kScenePathTraceCacheModeNrc;
        const Format historyFormat = exportDenoiserGuides(context.properties()) || nrcHistory
            ? Format::Rgba16Sfloat
            : Format::Rgba8Unorm;
        const TextureDesc historyDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled |
                TextureUsageBits::Storage |
                TextureUsageBits::TransferSource,
            .format = historyFormat,
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        };
        std::string historyName = historyNameForContext(context);
        if (nrcHistory) {
            // Distinct history slot: NRC mode accumulates linear HDR instead
            // of tonemapped colors.
            historyName += ".hdr";
        }
        Result result = history->ensureTexture(
            historyName,
            historyDesc,
            TextureViewDesc{.format = historyFormat});
        if (!result) {
            return result;
        }

        HistoryTextureRef current = history->texture(historyName, HistorySlot::Current);
        HistoryTextureRef previous = history->texture(historyName, HistorySlot::Previous);
        if (current.texture == nullptr || current.view == nullptr || previous.texture == nullptr || previous.view == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        result = history->transitionTexture(
            context.commandBuffer(),
            historyName,
            HistorySlot::Current,
            ResourceState::General);
        if (!result) {
            return result;
        }
        result = history->transitionTexture(
            context.commandBuffer(),
            historyName,
            HistorySlot::Previous,
            ResourceState::General);
        if (!result) {
            return result;
        }

        outCurrentView = current.view;
        outPreviousView = previous.view;
        historyCurrentTexture_ = current.texture;

        if (previous.valid && !resetAccumulation_) {
            ++accumulationFrame_;
            push.hasHistory = 1;
        } else {
            accumulationFrame_ = 0;
            push.hasHistory = 0;
        }
        resetAccumulation_ = false;
        push.accumulationFrame = accumulationFrame_;
        return {};
    }

    static uint32_t cacheModeFromProperties(const RenderGraphProperties& properties)
    {
        const std::string mode = stringProperty(properties, "cacheMode", "off");
        if (mode == "sharc" || mode == "SHaRC") {
            return kScenePathTraceCacheModeSharc;
        }
        if (mode == "nrc" || mode == "NRC") {
            return kScenePathTraceCacheModeNrc;
        }
        return kScenePathTraceCacheModeOff;
    }

    void clearPrograms()
    {
        for (ComputeProgram& program : programs_) {
            program.clear();
        }
        sharcClearProgram_.clear();
        sharcResolveProgram_.clear();
        tonemapProgram_.clear();
    }

    Result ensureCacheParamsBuffer(Device& device)
    {
        if (cacheParamsBuffer_ != nullptr) {
            return {};
        }
        BufferDesc desc{
            .size = sizeof(ScenePathTraceCacheParams),
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload,
        };
        std::unique_ptr<Buffer> buffer;
        Result result = device.createBuffer(desc, buffer);
        if (!result || buffer == nullptr) {
            spdlog::warn("[ScenePathTracePass] failed to create cache parameter buffer: {}",
                result ? "null buffer" : resultToString(result));
            return result ? makeError(Error::Failure) : result;
        }
        cacheParamsBuffer_ = std::move(buffer);
        return {};
    }

    Result writeCacheParamsBuffer(CommandBuffer& commandBuffer, const ScenePathTraceCacheParams& params)
    {
        if (cacheParamsBuffer_ == nullptr) {
            return makeError(Error::Failure);
        }
        void* mapped = cacheParamsBuffer_->map();
        if (mapped == nullptr) {
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, &params, sizeof(params));
        cacheParamsBuffer_->flush(0, sizeof(params));
        cacheParamsBuffer_->unmap();
        (void)commandBuffer;
        return {};
    }

    Result ensureSharcBuffers(Device& device, uint32_t entryCount)
    {
        if (sharcHashEntriesBuffer_ != nullptr && sharcAccumulationBuffer_ != nullptr &&
            sharcResolvedBuffer_ != nullptr && entryCount == sharcEntryCount_) {
            return {};
        }

        struct SharcBufferDesc {
            uint64_t elementSize;
            std::unique_ptr<Buffer>* target;
            const char* label;
        };
        const std::array<SharcBufferDesc, 3> descs{
            SharcBufferDesc{sizeof(uint64_t), &sharcHashEntriesBuffer_, "hash entries"},
            SharcBufferDesc{sizeof(uint32_t) * 4, &sharcAccumulationBuffer_, "accumulation"},
            SharcBufferDesc{sizeof(uint32_t) * 4, &sharcResolvedBuffer_, "resolved"},
        };
        for (const SharcBufferDesc& desc : descs) {
            BufferDesc bufferDesc{
                .size = static_cast<uint64_t>(entryCount) * desc.elementSize,
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::Device,
            };
            std::unique_ptr<Buffer> buffer;
            Result result = device.createBuffer(bufferDesc, buffer);
            if (!result || buffer == nullptr) {
                spdlog::warn("[ScenePathTracePass] failed to create SHaRC {} buffer: {}",
                    desc.label,
                    result ? "null buffer" : resultToString(result));
                for (const SharcBufferDesc& cleanup : descs) {
                    cleanup.target->reset();
                }
                sharcEntryCount_ = 0;
                return result ? makeError(Error::Failure) : result;
            }
            *desc.target = std::move(buffer);
        }
        sharcEntryCount_ = entryCount;
        sharcClearPending_ = true;
        return {};
    }

    Result executeSharcFrame(
        RenderGraphExecutionContext& context,
        ScenePathTracePush& push,
        const std::vector<ComputeDispatchBinding>& baseBindings,
        ComputeProgram& queryProgram,
        ComputeProgram& updateProgram)
    {
        const uint32_t entriesLog2 = uintProperty(
            context.properties(),
            "sharc.entriesLog2",
            kSharcDefaultEntriesLog2,
            kSharcMinEntriesLog2,
            kSharcMaxEntriesLog2);
        const uint32_t entryCount = 1u << entriesLog2;

        Result result = ensureSharcBuffers(*device_, entryCount);
        if (!result) {
            return result;
        }

        // Rebuild the cache whenever the scene, environment or capacity
        // changes; SHaRC handles camera movement through level blending.
        const uint64_t sharcRevision = sceneResourceRevision_ ^
            (environmentResourceRevision_ * 0x9e3779b97f4a7c15ull) ^
            (environmentSettingsRevision_ * 0xbf58476d1ce4e5b9ull) ^
            (static_cast<uint64_t>(entryCount) << 8u);
        if (sharcRevision != sharcResourcesRevision_) {
            sharcResourcesRevision_ = sharcRevision;
            sharcClearPending_ = true;
        }

        CommandBuffer& commandBuffer = context.commandBuffer();

        ScenePathTraceCacheParams params;
        copyFloat4(push.eye, params.sharcCameraPosition);
        copyFloat4(push.previousEye, params.sharcCameraPositionPrev);
        params.sharcEntriesNum = sharcEntryCount_;
        params.frameIndex = push.accumulationFrame;
        params.cacheMode = kScenePathTraceCacheModeSharc;
        params.width = push.width;
        params.height = push.height;
        const float sceneScaleSetting = floatProperty(context.properties(), "sharc.sceneScale", 0.0f);
        const float autoSceneScale = std::clamp(sceneResources_.bounds().radius() * 10.0f, 5.0f, 200.0f);
        params.sharcSceneScale = sceneScaleSetting > 0.0f ? sceneScaleSetting : autoSceneScale;
        params.sharcUpdateStride = uintProperty(
            context.properties(), "sharc.updateStride", kSharcDefaultUpdateStride, 1, 16);

        if (sharcClearPending_) {
            sharcClearPending_ = false;
            SceneSharcMaintenancePush clearPush;
            clearPush.entriesNum = sharcEntryCount_;
            result = dispatchSharcMaintenance(
                commandBuffer,
                sharcClearProgram_,
                clearPush,
                (sharcEntryCount_ + kSharcMaintenanceBlockSize - 1u) / kSharcMaintenanceBlockSize);
            if (!result) {
                return result;
            }
            barrierBuffers(commandBuffer, {sharcHashEntriesBuffer_.get(), sharcAccumulationBuffer_.get(), sharcResolvedBuffer_.get()});
        }

        result = writeCacheParamsBuffer(commandBuffer, params);
        if (!result) {
            return result;
        }

        // SHaRC update: sparse tracing over a stride x stride pixel block.
        std::vector<ComputeDispatchBinding> updateBindings = baseBindings;
        appendSharcDispatchBindings(updateBindings);
        const uint32_t stride = std::max(params.sharcUpdateStride, 1u);
        const uint32_t updateWidth = (push.width + stride - 1u) / stride;
        const uint32_t updateHeight = (push.height + stride - 1u) / stride;
        result = updateProgram.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = updateBindings.data(),
            .bindingCount = static_cast<uint32_t>(updateBindings.size()),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (updateWidth + 7) / 8,
            .groupCountY = (updateHeight + 7) / 8,
            .groupCountZ = 1,
        });
        if (!result) {
            return result;
        }
        barrierBuffers(commandBuffer, {sharcHashEntriesBuffer_.get(), sharcAccumulationBuffer_.get(), sharcResolvedBuffer_.get()});

        // SHaRC resolve: combine per-frame accumulation with previous data.
        SceneSharcMaintenancePush resolvePush;
        copyFloat4(push.eye, resolvePush.cameraPosition);
        copyFloat4(push.previousEye, resolvePush.cameraPositionPrev);
        resolvePush.sceneScale = params.sharcSceneScale;
        resolvePush.entriesNum = sharcEntryCount_;
        resolvePush.accumulationFrameNum = uintProperty(
            context.properties(),
            "sharc.maxAccumulatedFrames",
            kSharcDefaultMaxAccumulatedFrames,
            1,
            1024);
        resolvePush.staleFrameNumMax = uintProperty(
            context.properties(),
            "sharc.staleFrameNum",
            kSharcDefaultStaleFrameNum,
            8,
            1024);
        resolvePush.frameIndex = push.accumulationFrame;
        result = dispatchSharcMaintenance(
            commandBuffer,
            sharcResolveProgram_,
            resolvePush,
            (sharcEntryCount_ + kSharcMaintenanceBlockSize - 1u) / kSharcMaintenanceBlockSize);
        if (!result) {
            return result;
        }
        barrierBuffers(commandBuffer, {sharcHashEntriesBuffer_.get(), sharcAccumulationBuffer_.get(), sharcResolvedBuffer_.get()});

        // SHaRC render/query at full resolution with early termination.
        std::vector<ComputeDispatchBinding> queryBindings = baseBindings;
        appendSharcDispatchBindings(queryBindings);
        return queryProgram.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = queryBindings.data(),
            .bindingCount = static_cast<uint32_t>(queryBindings.size()),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (push.width + 7) / 8,
            .groupCountY = (push.height + 7) / 8,
            .groupCountZ = 1,
        });
    }

    Result dispatchSharcMaintenance(
        CommandBuffer& commandBuffer,
        ComputeProgram& program,
        const SceneSharcMaintenancePush& maintenancePush,
        uint32_t groupCount)
    {
        const std::array<ComputeDispatchBinding, 4> bindings{
            ComputeDispatchBinding{.binding = 0, .buffer = sharcHashEntriesBuffer_.get()},
            ComputeDispatchBinding{.binding = 1, .buffer = sharcAccumulationBuffer_.get()},
            ComputeDispatchBinding{.binding = 2, .buffer = sharcResolvedBuffer_.get()},
            ComputeDispatchBinding{.binding = 3, .buffer = cacheParamsBuffer_.get()},
        };
        return program.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = bindings.data(),
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pushData = &maintenancePush,
            .pushDataSize = sizeof(maintenancePush),
            .groupCountX = std::max(groupCount, 1u),
            .groupCountY = 1,
            .groupCountZ = 1,
        });
    }

    void appendSharcDispatchBindings(std::vector<ComputeDispatchBinding>& bindings) const
    {
        bindings.push_back(ComputeDispatchBinding{
            .binding = kScenePathTraceCacheParamsBinding,
            .buffer = cacheParamsBuffer_.get(),
        });
        bindings.push_back(ComputeDispatchBinding{
            .binding = kScenePathTraceSharcHashEntriesBinding,
            .buffer = sharcHashEntriesBuffer_.get(),
        });
        bindings.push_back(ComputeDispatchBinding{
            .binding = kScenePathTraceSharcAccumulationBinding,
            .buffer = sharcAccumulationBuffer_.get(),
        });
        bindings.push_back(ComputeDispatchBinding{
            .binding = kScenePathTraceSharcResolvedBinding,
            .buffer = sharcResolvedBuffer_.get(),
        });
    }

    static void barrierBuffers(
        CommandBuffer& commandBuffer,
        std::initializer_list<Buffer*> buffers)
    {
        std::vector<BufferBarrierDesc> barriers;
        barriers.reserve(buffers.size());
        for (Buffer* buffer : buffers) {
            if (buffer == nullptr) {
                continue;
            }
            barriers.push_back(BufferBarrierDesc{
                .buffer = buffer,
                .before = ResourceState::General,
                .after = ResourceState::General,
            });
        }
        if (!barriers.empty()) {
            commandBuffer.barrier(BarrierDesc{
                .buffers = barriers.data(),
                .bufferCount = static_cast<uint32_t>(barriers.size()),
            });
        }
    }

#if METALLIC_HAS_NRC
    static NrcResolveMode nrcResolveModeFromProperties(const RenderGraphProperties& properties)
    {
        const std::string mode = stringProperty(properties, "nrc.resolveMode", "add");
        if (mode == "replace") {
            return NrcResolveMode::ReplaceOutputWithQueryResult;
        }
        if (mode == "heatmap") {
            return NrcResolveMode::TrainingBounceHeatMap;
        }
        if (mode == "queryIndex") {
            return NrcResolveMode::QueryIndex;
        }
        if (mode == "cacheView") {
            return NrcResolveMode::DirectCacheView;
        }
        return NrcResolveMode::AddQueryResultToOutput;
    }

    Result executeNrcFrame(
        RenderGraphExecutionContext& context,
        ScenePathTracePush& push,
        const std::vector<ComputeDispatchBinding>& baseBindings,
        ComputeProgram& updateProgram,
        ComputeProgram& queryProgram,
        TextureView* historyCurrentView)
    {
        if (device_ == nullptr || graphicsQueue_ == nullptr || historyCurrentView == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        CommandBuffer& commandBuffer = context.commandBuffer();

        // NRC requires EndFrame after submission; defer it to the next frame's
        // execute, when the previous command buffer is guaranteed submitted.
        if (nrcEndFramePending_) {
            nrcEndFramePending_ = false;
            Result endResult = nrc_.endFrame(*graphicsQueue_);
            if (!endResult) {
                return endResult;
            }
        }
        if (!nrc_.valid()) {
            std::string nrcLog;
            Result initResult = nrc_.initialize(*device_, nrcLog);
            if (!initResult) {
                spdlog::warn("[ScenePathTracePass] NRC initialization failed: {}", nrcLog);
                return initResult;
            }
        }

        const scene::Bounds& bounds = sceneResources_.bounds();
        nrc::ContextSettings settings{};
        settings.learnIrradiance = false;
        settings.includeDirectLighting = false;
        settings.requestReset =
            sceneResourceRevision_ != nrcSceneRevision_ ||
            environmentResourceRevision_ != nrcEnvironmentRevision_;
        settings.sceneBoundsMin = nrc_float3{bounds.min.x, bounds.min.y, bounds.min.z};
        settings.sceneBoundsMax = nrc_float3{bounds.max.x, bounds.max.y, bounds.max.z};
        settings.smallestResolvableFeatureSize = std::max(bounds.radius() * 0.001f, 0.001f);
        settings.frameDimensions = nrc_uint2{context.width(), context.height()};
        const nrc_uint2 idealTraining =
            nrc::ComputeIdealTrainingDimensions(settings.frameDimensions, 4);
        settings.trainingDimensions = nrc_uint2{
            std::min(idealTraining.x, context.width()),
            std::min(idealTraining.y, context.height())};
        settings.samplesPerPixel = 1;
        settings.maxPathVertices = kNrcMaxPathVertices;

        const bool reconfigure =
            !nrcConfigured_ || settings != nrcContextSettings_ || settings.requestReset;
        if (reconfigure) {
            std::string configureLog;
            Result configureResult = nrc_.configure(settings, *device_, configureLog);
            if (!configureResult) {
                spdlog::warn("[ScenePathTracePass] NRC configure failed: {}", configureLog);
                return configureResult;
            }
            nrcContextSettings_ = settings;
            nrcConfigured_ = true;
            nrcSceneRevision_ = sceneResourceRevision_;
            nrcEnvironmentRevision_ = environmentResourceRevision_;
        }

        nrc::FrameSettings frameSettings{};
        frameSettings.maxExpectedAverageRadianceValue =
            floatProperty(context.properties(), "nrc.maxExpectedRadiance", 1.0f);
        frameSettings.resolveMode = nrcResolveModeFromProperties(context.properties());
        Result result = nrc_.beginFrame(commandBuffer, frameSettings);
        if (!result) {
            return result;
        }

        ::NrcConstants nrcConstants;
        result = nrc_.populateShaderConstants(nrcConstants);
        if (!result) {
            return result;
        }

        ScenePathTraceCacheParams params;
        copyFloat4(push.eye, params.sharcCameraPosition);
        copyFloat4(push.previousEye, params.sharcCameraPositionPrev);
        params.sharcEntriesNum = 0;
        params.frameIndex = push.accumulationFrame;
        params.cacheMode = kScenePathTraceCacheModeNrc;
        params.width = push.width;
        params.height = push.height;
        params.trainingWidth = nrcContextSettings_.trainingDimensions.x;
        params.trainingHeight = nrcContextSettings_.trainingDimensions.y;
        params.nrcFrameDimensions[0] = nrcConstants.frameDimensions.x;
        params.nrcFrameDimensions[1] = nrcConstants.frameDimensions.y;
        params.nrcTrainingDimensions[0] = nrcConstants.trainingDimensions.x;
        params.nrcTrainingDimensions[1] = nrcConstants.trainingDimensions.y;
        params.nrcScenePosScale[0] = nrcConstants.scenePosScale.x;
        params.nrcScenePosScale[1] = nrcConstants.scenePosScale.y;
        params.nrcScenePosScale[2] = nrcConstants.scenePosScale.z;
        params.nrcSamplesPerPixel = nrcConstants.samplesPerPixel;
        params.nrcScenePosBias[0] = nrcConstants.scenePosBias.x;
        params.nrcScenePosBias[1] = nrcConstants.scenePosBias.y;
        params.nrcScenePosBias[2] = nrcConstants.scenePosBias.z;
        params.nrcMaxPathVertices = nrcConstants.maxPathVertices;
        params.nrcLearnIrradiance = nrcConstants.learnIrradiance;
        params.nrcRadianceCacheDirect = nrcConstants.radianceCacheDirect;
        params.nrcRadianceUnpackMultiplier = nrcConstants.radianceUnpackMultiplier;
        params.nrcResolveMode = static_cast<int32_t>(nrcConstants.resolveMode);
        params.nrcEnableTerminationHeuristic = nrcConstants.enableTerminationHeuristic;
        params.nrcSkipDeltaVertices = nrcConstants.skipDeltaVertices;
        params.nrcTerminationHeuristicThreshold = nrcConstants.terminationHeuristicThreshold;
        params.nrcTrainingTerminationHeuristicThreshold = nrcConstants.trainingTerminationHeuristicThreshold;
        params.nrcProportionUnbiased = nrcConstants.proportionUnbiased;
        result = writeCacheParamsBuffer(commandBuffer, params);
        if (!result) {
            return result;
        }

        std::vector<ComputeDispatchBinding> traceBindings = baseBindings;
        traceBindings.push_back(ComputeDispatchBinding{
            .binding = kScenePathTraceCacheParamsBinding,
            .buffer = cacheParamsBuffer_.get(),
        });
        static constexpr nrc::BufferIdx kTraceBuffers[] = {
            nrc::BufferIdx::QueryPathInfo,
            nrc::BufferIdx::TrainingPathInfo,
            nrc::BufferIdx::TrainingPathVertices,
            nrc::BufferIdx::QueryRadianceParams,
            nrc::BufferIdx::Counter,
        };
        static constexpr uint32_t kTraceBindings[] = {
            kScenePathTraceNrcQueryPathInfoBinding,
            kScenePathTraceNrcTrainingPathInfoBinding,
            kScenePathTraceNrcTrainingPathVerticesBinding,
            kScenePathTraceNrcQueryRadianceParamsBinding,
            kScenePathTraceNrcCountersBinding,
        };
        for (size_t index = 0; index < std::size(kTraceBuffers); ++index) {
            traceBindings.push_back(ComputeDispatchBinding{
                .binding = kTraceBindings[index],
                .buffer = nrc_.buffer(static_cast<uint32_t>(kTraceBuffers[index])),
            });
        }

        // NRC update pass at training resolution writes training data only.
        const uint32_t trainingWidth = std::max(params.trainingWidth, 1u);
        const uint32_t trainingHeight = std::max(params.trainingHeight, 1u);
        result = updateProgram.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = traceBindings.data(),
            .bindingCount = static_cast<uint32_t>(traceBindings.size()),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (trainingWidth + 7) / 8,
            .groupCountY = (trainingHeight + 7) / 8,
            .groupCountZ = 1,
        });
        if (!result) {
            return result;
        }

        // NRC query pass at full resolution; linear HDR output into history.
        result = queryProgram.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = traceBindings.data(),
            .bindingCount = static_cast<uint32_t>(traceBindings.size()),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (push.width + 7) / 8,
            .groupCountY = (push.height + 7) / 8,
            .groupCountZ = 1,
        });
        if (!result) {
            return result;
        }

        // The path tracer wrote the training/query records; make them visible
        // to the NRC library kernels.
        barrierBuffers(commandBuffer, {cacheParamsBuffer_.get()});
        for (size_t index = 0; index < std::size(kTraceBuffers); ++index) {
            Buffer* buffer = nrc_.buffer(static_cast<uint32_t>(kTraceBuffers[index]));
            if (buffer != nullptr) {
                barrierBuffers(commandBuffer, {buffer});
            }
        }

        result = nrc_.queryAndTrain(commandBuffer, nullptr);
        if (!result) {
            return result;
        }
        result = nrc_.resolve(commandBuffer, *historyCurrentView);
        if (!result) {
            return result;
        }
        nrcEndFramePending_ = true;

        // Resolve wrote linear HDR into the history texture; tonemap it into
        // the displayable color output.
        Texture* historyTexture = historyCurrentTexture_;
        if (historyTexture != nullptr) {
            TextureBarrierDesc historyBarrier{
                .texture = historyTexture,
                .before = ResourceState::General,
                .after = ResourceState::General,
            };
            commandBuffer.barrier(BarrierDesc{
                .textures = &historyBarrier,
                .textureCount = 1,
            });
        }
        ScenePathTraceTonemapPush tonemapPush{
            .width = push.width,
            .height = push.height,
        };
        const std::array<ComputeDispatchBinding, 2> tonemapBindings{
            ComputeDispatchBinding{.binding = 0, .textureView = historyCurrentView},
            ComputeDispatchBinding{.binding = 1, .textureView = context.outputTexture("color").view()},
        };
        return tonemapProgram_.dispatch(ComputeDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = tonemapBindings.data(),
            .bindingCount = static_cast<uint32_t>(tonemapBindings.size()),
            .pushData = &tonemapPush,
            .pushDataSize = sizeof(tonemapPush),
            .groupCountX = (push.width + 7) / 8,
            .groupCountY = (push.height + 7) / 8,
            .groupCountZ = 1,
        });
    }
#endif

    static uint32_t uintProperty(
        const RenderGraphProperties& properties,
        const char* key,
        uint32_t fallback,
        uint32_t minimum,
        uint32_t maximum)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number()) {
            return fallback;
        }
        uint32_t value = fallback;
        if (iter->is_number_unsigned()) {
            value = iter->get<uint32_t>();
        } else if (iter->is_number_integer()) {
            const int64_t signedValue = iter->get<int64_t>();
            value = signedValue > 0 ? static_cast<uint32_t>(signedValue) : minimum;
        } else {
            value = static_cast<uint32_t>(std::max(iter->get<float>(), static_cast<float>(minimum)));
        }
        return std::clamp(value, minimum, maximum);
    }

    static bool boolProperty(const RenderGraphProperties& properties, const char* key, bool fallback)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_boolean()) {
            return fallback;
        }
        return iter->get<bool>();
    }

    static float floatProperty(const RenderGraphProperties& properties, const char* key, float fallback)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number()) {
            return fallback;
        }
        return finiteOr(iter->get<float>(), fallback);
    }

    static std::string stringProperty(
        const RenderGraphProperties& properties,
        const char* key,
        std::string_view fallback)
    {
        if (!properties.is_object()) {
            return std::string(fallback);
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_string()) {
            return std::string(fallback);
        }
        return iter->get<std::string>();
    }

    static bool useOpenPBRBsdf(const RenderGraphProperties& properties)
    {
        const std::string bsdf = stringProperty(properties, "bsdf", "standard");
        return bsdf == "openpbr" || bsdf == "OpenPBR";
    }

    static std::string historyNameForContext(const RenderGraphExecutionContext& context)
    {
        std::string name(kScenePathTraceHistoryPrefix);
        name += context.passName();
        name += ".accumulation";
        return name;
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

    static float cameraFloat(const RenderGraphProperties* camera, const char* key, float fallback)
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

    static void writeParamVec3(const float3& value, float out[4], float w)
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = w;
    }

    static void copyFloat4(const float source[4], float target[4])
    {
        std::copy(source, source + 4, target);
    }

    static ScenePathTraceCameraSnapshot cameraSnapshotFromPush(const ScenePathTracePush& push)
    {
        ScenePathTraceCameraSnapshot snapshot;
        copyFloat4(push.eye, snapshot.eye);
        copyFloat4(push.center, snapshot.center);
        copyFloat4(push.upProjection, snapshot.upProjection);
        copyFloat4(push.viewport, snapshot.viewport);
        copyFloat4(push.clipOrtho, snapshot.clipOrtho);
        return snapshot;
    }

    static void applyPreviousCameraSnapshot(
        const ScenePathTraceCameraSnapshot& snapshot,
        ScenePathTracePush& push)
    {
        copyFloat4(snapshot.eye, push.previousEye);
        copyFloat4(snapshot.center, push.previousCenter);
        copyFloat4(snapshot.upProjection, push.previousUpProjection);
        copyFloat4(snapshot.viewport, push.previousViewport);
        copyFloat4(snapshot.clipOrtho, push.previousClipOrtho);
    }

    static void buildPush(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        const scene::Bounds& drawBounds,
        const EnvironmentSettings& environment,
        bool environmentMapAvailable,
        ScenePathTracePush& outPush)
    {
        outPush = ScenePathTracePush{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 50.0f),
            1.0f,
            179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + radius
                : radius * 2.5f,
            0.05f);
        const float3 defaultEye(center.x, center.y + radius * 0.12f, center.z + defaultDistance);
        const float3 eye = cameraVec3(cameraProperties, "eye", defaultEye);
        const float3 target = cameraVec3(cameraProperties, "center", center);
        const float3 up = cameraVec3(cameraProperties, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.001f), 0.0001f);
        const float zFar = std::max(
            cameraFloat(cameraProperties, "zfar", defaultDistance + radius * 4.0f),
            zNear + 0.001f);
        const float cameraDistance = std::max(length(eye - target), 0.001f);
        const float defaultOrthoHeight = std::max(2.0f * cameraDistance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float orthoHeight = std::max(
            cameraFloat(cameraProperties, "orthoHeight", defaultOrthoHeight),
            0.0001f);

        writeParamVec3(eye, outPush.eye, 0.0f);
        writeParamVec3(target, outPush.center, 0.0f);
        writeParamVec3(up, outPush.upProjection, cameraIsOrthographic(cameraProperties) ? 1.0f : 0.0f);
        outPush.viewport[0] = aspect;
        outPush.viewport[1] = static_cast<float>(width);
        outPush.viewport[2] = static_cast<float>(height);
        outPush.viewport[3] = fovRadians;
        outPush.clipOrtho[0] = zNear;
        outPush.clipOrtho[1] = zFar;
        outPush.clipOrtho[2] = orthoHeight;
        outPush.clipOrtho[3] = 0.0f;
        outPush.width = width;
        outPush.height = height;
        outPush.maxDepth = uintProperty(properties, "maxDepth", kDefaultPathTraceMaxDepth, 1, kMaxPathTraceMaxDepth);
        outPush.samples = uintProperty(properties, "samples", kDefaultPathTraceSamples, 1, kMaxPathTraceSamples);
        outPush.bitangentFlip = metallic::render::builtin_pass::boolProperty(&properties, "flipBitangent", false)
            ? -1.0f
            : 1.0f;

        outPush.environmentIntensity = std::max(environment.intensity, 0.0f);
        outPush.environmentRotationRadians = environment.rotationDegrees * (kPi / 180.0f);
        outPush.environmentMode = kScenePathTraceEnvironmentModeProcedural;
        outPush.environmentVisible = environment.visible ? 1u : 0u;
        if (!environment.enabled) {
            outPush.environmentMode = kScenePathTraceEnvironmentModeDisabled;
        } else if (environmentMapAvailable) {
            outPush.environmentMode = kScenePathTraceEnvironmentModeMap;
        }
    }

    ScenePathTraceResources sceneResources_;
    SceneResourceManager* sceneResourceManager_ = nullptr;
    Device* device_ = nullptr;
    Queue* graphicsQueue_ = nullptr;
    OpenPBRLutResources openPBRLuts_;
    std::array<ComputeProgram, static_cast<size_t>(PathTracePermutation::Count)> programs_;
    ComputeProgram sharcClearProgram_;
    ComputeProgram sharcResolveProgram_;
    ComputeProgram tonemapProgram_;
    std::string compiledShaderKey_;
    uint32_t cacheMode_ = kScenePathTraceCacheModeOff;
    std::unique_ptr<Buffer> cacheParamsBuffer_;
    std::unique_ptr<Buffer> sharcHashEntriesBuffer_;
    std::unique_ptr<Buffer> sharcAccumulationBuffer_;
    std::unique_ptr<Buffer> sharcResolvedBuffer_;
    uint32_t sharcEntryCount_ = 0;
    uint64_t sharcResourcesRevision_ = 0;
    bool sharcClearPending_ = false;
#if METALLIC_HAS_NRC
    vulkan::NrcIntegration nrc_;
    nrc::ContextSettings nrcContextSettings_{};
    bool nrcConfigured_ = false;
    bool nrcEndFramePending_ = false;
    uint64_t nrcSceneRevision_ = 0;
    uint64_t nrcEnvironmentRevision_ = 0;
#endif
    Texture* historyCurrentTexture_ = nullptr;
    uint64_t sceneResourceRevision_ = 0;
    uint64_t environmentResourceRevision_ = 0;
    uint64_t environmentSettingsRevision_ = 0;
    uint32_t accumulationFrame_ = 0;
    ScenePathTraceCameraSnapshot previousCamera_;
    uint32_t previousCameraWidth_ = 0;
    uint32_t previousCameraHeight_ = 0;
    bool hasPreviousCamera_ = false;
    bool resetAccumulation_ = false;
};

} // namespace

std::unique_ptr<RenderGraphPass> createScenePathTracePass()
{
    return std::make_unique<ScenePathTracePass>();
}

} // namespace metallic::render::builtin_pass
