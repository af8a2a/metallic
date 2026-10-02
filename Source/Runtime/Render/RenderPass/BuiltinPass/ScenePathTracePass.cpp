#include "Runtime/Render/Core/DeferredShadingParameters.h"
#include "Runtime/Render/Core/RealtimeLightingParameters.h"
#include "Runtime/Render/Core/PathTraceGuidesInlineParameters.h"
#include "Runtime/Render/Core/NRCTraceParameters.h"
#include "Runtime/Render/Core/SharcTraceParameters.h"
#include "Runtime/Render/Core/PathTraceInlineParameters.h"
#include "Runtime/Render/Core/PathTraceParameters.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/PathTraceStageParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNRCWrapper.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/SceneLightResources.h"
#include "Runtime/Render/MaterialBinning.h"
#include "Runtime/Render/Material/MaterialExecutable.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Render/RenderPass/BuiltinPass/ScreenSpaceShadowPassCommon.h"
#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include "openpbr_data_constants.h"

#include <chrono>

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
constexpr uint32_t kNRCMaxPathVertices = 8;

enum class PathTracePermutation : uint32_t {
    Base = 0,
    SharcUpdate,
    SharcQuery,
    NRCUpdate,
    NRCQuery,
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
    case PathTracePermutation::NRCUpdate:
        return "nrc-update";
    case PathTracePermutation::NRCQuery:
        return "nrc-query";
    default:
        return "?";
    }
}

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
constexpr uint32_t kDLSSRRAlbedoBinding = 14;
constexpr uint32_t kDLSSRRSpecularAlbedoBinding = 15;
constexpr uint32_t kDLSSRRNormalRoughnessBinding = 16;
constexpr uint32_t kDLSSRRMotionVectorsBinding = 17;
constexpr uint32_t kDLSSRRLinearDepthBinding = 18;
constexpr uint32_t kDLSSRRSpecularHitDistanceBinding = 19;
constexpr uint32_t kDLSSDepthBinding = 20;
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
    struct UploadState {
        ResourceState state = ResourceState::Undefined;
        bool uploaded = false;
    };
    std::unique_ptr<Buffer> uploadBuffer;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    std::shared_ptr<UploadState> uploadState = std::make_shared<UploadState>();
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
};

class OpenPBRLutResources final {
public:
    Result<> prepare(Device& device, std::string& log)
    {
        if (valid()) {
            return {};
        }

        clear();
        Result<> result = createScalarLut(
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

    Result<> upload(CommandBuffer& commandBuffer)
    {
        for (OpenPBRLutTexture& texture : lut2D_) {
            Result<> result = uploadTexture(commandBuffer, texture);
            if (!result) {
                return result;
            }
        }
        for (OpenPBRLutTexture& texture : lut3D_) {
            Result<> result = uploadTexture(commandBuffer, texture);
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
    static Result<> createScalarLut(
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

    static Result<> createLtcLut(Device& device, OpenPBRLutTexture& outTexture, std::string& log)
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

    static Result<> createRgbaLutTexture(
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
        Result<> result = device.createBuffer(BufferDesc{
                .size = byteSize,
                .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload,
            }).transform([&](auto rhiValue) { outTexture.uploadBuffer = std::move(rhiValue); });
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
        outTexture.uploadBuffer->flush({0, byteSize});
        outTexture.uploadBuffer->unmap();

        result = device.createTexture(TextureDesc{
                .type = depth > 1 ? TextureType::Texture3D : TextureType::Texture2D,
                .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::RGBA32Sfloat,
                .width = width,
                .height = height,
                .depth = depth,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            }).transform([&](auto rhiValue) { outTexture.texture = std::move(rhiValue); });
        if (!result || outTexture.texture == nullptr) {
            log += resultMessage(std::string("createTexture(") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = device.createTextureView(*outTexture.texture,
            TextureViewDesc{
                .format = Format::RGBA32Sfloat,
                .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
            }).transform([&](auto rhiValue) { outTexture.view = std::move(rhiValue); });
        if (!result || outTexture.view == nullptr) {
            log += resultMessage(std::string("createTextureView(") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    static Result<> uploadTexture(CommandBuffer& commandBuffer, OpenPBRLutTexture& texture)
    {
        if (texture.uploadState->uploaded) {
            return {};
        }
        if (texture.uploadBuffer == nullptr || texture.texture == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        using namespace detail;
        auto upload = texture.uploadBuffer->slice({0, texture.uploadBuffer->desc().size});
        if (!upload) { return makeError(upload.error()); }
        const std::array resources{
            GraphAccessResource{RenderGraphResourceType::Texture2D, texture.uploadState->state},
            GraphAccessResource{RenderGraphResourceType::Buffer, ResourceState::General,
                {PipelineStageBits::Host, AccessBits::HostWrite}}};
        const std::array bindings{GraphAccessBinding{.texture = texture.texture.get()},
            GraphAccessBinding{.buffer = *upload}};
        const std::array accesses{
            GraphAccessPass{.uses = {{0, ResourceState::TransferDestination,
                {PipelineStageBits::Transfer, AccessBits::TransferWrite}, true},
                {1, ResourceState::TransferSource, {PipelineStageBits::Transfer, AccessBits::TransferRead}, false}}},
            GraphAccessPass{.uses = {{0, ResourceState::ShaderRead,
                {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}, false}}}};
        auto plan = buildGraphAccessPlan(resources, accesses);
        if (!plan) { return makeError(plan.error()); }
        auto result = commandBuffer.retainResource(texture.view->retainTexture());
        if (!result) { return result; }
        const auto state = texture.uploadState;
        result = commandBuffer.addSubmissionTransaction(std::make_shared<SubmissionTransaction>([] {}, [state] {
            state->state = ResourceState::Undefined;
            state->uploaded = false;
        }));
        if (!result) { return result; }
        result = recordGraphAccessBarriers(commandBuffer, plan->passes[0], bindings);
        if (!result) { return result; }

        commandBuffer.copyBufferToTexture(BufferTextureCopyDesc{
            .buffer = texture.uploadBuffer.get(),
            .texture = texture.texture.get(),
            .width = texture.width,
            .height = texture.height,
            .depth = texture.depth,
            .mipLevel = 0,
            .baseLayer = 0,
        });

        result = recordGraphAccessBarriers(commandBuffer, plan->passes[1], bindings);
        if (!result) { return result; }
        state->state = ResourceState::ShaderRead;
        state->uploaded = true;
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
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext& context) const override
    {
        if (context.runtimeScene && context.runtimeScene->hasStreamGeometry()) {
            return {.features = SceneResourceFeatureBits::Materials | SceneResourceFeatureBits::MaterialTextures, .textureFeedback = visibilityDeferred_};
        }
        return {.features = SceneResourceFeatureBits::Geometry | SceneResourceFeatureBits::Materials |
            SceneResourceFeatureBits::MaterialTextures | SceneResourceFeatureBits::StandardAccelerationStructure, .textureFeedback = visibilityDeferred_};
    }

    RenderGraphSceneDependency sceneDependency() const override
    {
        return visibilityDeferred_
            ? RenderGraphSceneDependency{RenderGraphSceneSource::Input, {"visibility", "depth", "rasterInfo"}}
            : RenderGraphSceneDependency{RenderGraphSceneSource::World};
    }

    explicit ScenePathTracePass(bool realtime = false, bool visibilityDeferred = false)
        : realtime_(realtime), visibilityDeferred_(visibilityDeferred) {}

    bool supportsFrameOverlap() const override
    {
        return cacheMode_ == kScenePathTraceCacheModeOff &&
            (!METALLIC_HAS_NRD || !visibilityDeferred_ || properties().value("lightingMode", "reference") != "realtime");
    }

    ~ScenePathTracePass() override
    {
#if METALLIC_HAS_NRC
        // The final submitted frame has no following execute() to finish it.
        if (*nrcEndFramePending_ && graphicsQueue_ != nullptr) {
            (void)nrc_.endFrame(*graphicsQueue_);
            if (device_ != nullptr) {
                (void)device_->waitIdle();
            }
        }
#endif
    }

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array deferredRequired{
            EnvironmentLightingSubsystem::kSubsystemId, GPUSceneSubsystem::kSubsystemId,
        };
        if (visibilityDeferred_) { return deferredRequired; }
        static constexpr std::array required{
            EnvironmentLightingSubsystem::kSubsystemId,
        };
        return required;
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        const bool exportGuides = exportDenoiserGuides(properties());
        RenderPassReflection reflection;
        reflection.addAccelerationStructureInput("accelerationStructure", "Optional graph-managed scene TLAS/PTLAS")
            .accelerationStructureRead().setOptional();
        if (visibilityDeferred_) {
            reflection.addTextureInput("visibility", "Resident GPUScene visibility IDs").sampledRead().format = Format::R32Uint;
            reflection.addTextureInput("depth", "Depth from the same visibility raster").sampledRead().format = Format::D32Sfloat;
            auto& domain = reflection.addTextureInput("domain", "Optional displaced barycentrics and geometric normal").sampledRead().setOptional();
            domain.format = Format::Unknown; // R32G32B32A32 when active, inexpensive dummy when disabled.
            reflection.addBufferInput("rasterInfo", "CPU-authored raster camera and scene identity")
                .buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo)).shaderRead();
            if (properties().value("lightingMode", "reference") == "realtime") {
                reflection.addTextureInput("shadow", "SIGMA-encoded visibility from RayTracedShadowPass")
                    .sampledRead().setOptional().format = Format::R8Unorm;
                reflection.addBufferInput("shadowParameters", "Matching shadow light and trace metadata")
                    .buffer(sizeof(ScreenSpaceShadowParameters), sizeof(ScreenSpaceShadowParameters)).shaderRead().setOptional();
            }
        }
        auto& color = reflection.addTextureOutput("color", visibilityDeferred_ ? "OpenPBR deferred physical HDR" :
            (realtime_ ? "Real-time physical lighting and SH GI" : "Path-traced glTF scene"))
            .storageReadWrite();
        color.colorEncoding = DisplayColorEncoding::SceneLinear;
        color.format = (exportGuides || (visibilityDeferred_ && boolProperty(properties(), "exportUpscalerGuides", false))) ? Format::RGBA16Sfloat :
                Format::RGBA32Sfloat;
        if (cacheModeFromProperties(properties()) == kScenePathTraceCacheModeNRC) {
            color.stageAccess(RenderGraphResourceAccess::TextureStorageReadWrite, RenderGraphPassKind::Unsafe);
        }
        if (visibilityDeferred_ && boolProperty(properties(), "exportUpscalerGuides", false)) {
            reflection.addTextureOutput("motionVectors", "Unjittered current-to-previous UV motion")
                .storageReadWrite().format = Format::RG16Sfloat;
            reflection.addTextureOutput("deviceDepth", "Raster hardware depth for DLSS-SR")
                .storageReadWrite().format = Format::R32Sfloat;
        }
        if (exportGuides) {
            reflection.addTextureOutput("albedo", "DLSS-RR diffuse albedo guide")
                .storageReadWrite()
                .format = Format::RGBA16Sfloat;
            reflection.addTextureOutput("specularAlbedo", "DLSS-RR specular albedo guide")
                .storageReadWrite()
                .format = Format::RGBA16Sfloat;
            reflection.addTextureOutput("normalRoughness", "DLSS-RR packed normal and roughness guide")
                .storageReadWrite()
                .format = Format::RGBA16Sfloat;
            reflection.addTextureOutput("motionVectors", "DLSS-RR motion vector guide")
                .storageReadWrite()
                .format = Format::RG16Sfloat;
            reflection.addTextureOutput("linearDepth", "DLSS-RR linear depth guide")
                .storageReadWrite()
                .format = Format::R32Sfloat;
            reflection.addTextureOutput("specularHitDistance", "DLSS-RR specular hit distance guide")
                .storageReadWrite()
                .format = Format::R32Sfloat;
            reflection.addTextureOutput("depth", "DLSS normalized hardware depth")
                .storageReadWrite()
                .format = Format::R32Sfloat;
        }
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        if (realtime_) {
            std::vector<RenderGraphRuntimeSetting> settings{
                runtimeBoolSetting("flipBitangent", "Flip Bitangent", false, true),
                runtimeBoolSetting("debugDisableShadows", "Disable Shadows", false, true),
            };
            if (visibilityDeferred_) {
                settings.erase(settings.begin()); // Deferred resolve always exports physical HDR.
                settings.push_back(runtimeEnumSetting("lightingMode", "Environment Lighting", "reference",
                    {{"SH + Filtered HDRI", "realtime"}, {"Sampled Reference", "reference"}}, true, true));
                auto guides = runtimeBoolSetting("exportUpscalerGuides", "Export DLSS-SR Guides", false, true);
                guides.rebuildGraph = true;
                settings.push_back(guides);
                settings.push_back(runtimeIntSetting("environmentSamples", "OpenPBR Environment Samples", 64, 1, 256));
                const auto shadowSettings = screenSpaceShadowRuntimeSettings(properties());
                settings.insert(settings.end(), shadowSettings.begin(), shadowSettings.end());
                auto binning = runtimeBoolSetting("materialBinning", "Wave32 Material Tile Classification", true, true);
                binning.rebuildGraph = true;
                settings.push_back(binning);
                auto pathTracing = runtimeBoolSetting("supplementaryPathTracing", "Supplementary Path Tracing", false, true);
                pathTracing.rebuildGraph = true;
                settings.push_back(pathTracing);
                if (boolProperty(properties(), "supplementaryPathTracing", false)) {
                    settings.push_back(runtimeIntSetting("transmissionSamples", "Transmission Samples", 2, 1, 16, true));
                    settings.push_back(runtimeIntSetting("transmissionDepth", "Transmission Max Depth", 8, 2, 16, true));
                }
                settings.push_back(runtimeBoolSetting("debugDisableTransmission", "Disable Transmission", false, true));
                settings.push_back(runtimeBoolSetting("debugDisableVolumeAttenuation", "Disable Volume Absorption", false, true));
                settings.push_back(runtimeBoolSetting("stochasticTextureFiltering", "Stochastic Texture Filtering", false, true));
                settings.push_back(runtimeBoolSetting("debugUseOpaqueShadows", "Use Opaque Shadows", false, true));
                settings.push_back(runtimeBoolSetting("accumulate", "Accumulate Lighting", true, true));
                settings.push_back(runtimeEnumSetting("debugView", "Surface Debug", "final",
                    {{"Final", "final"}, {"Base Color", "baseColor"}, {"Geometry Normal", "geometryNormal"},
                        {"Shading Normal", "shadingNormal"}, {"Tangent", "tangent"}, {"Material ID", "material"}}));
                return settings; // Camera is supplied by rasterInfo, not a second editable camera.
            }
            appendCameraRuntimeSettings(settings, {0.0f, 0.2f, 2.5f}, {0.0f, 0.0f, 0.0f}, 50.0f, true);
            return settings;
        }
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
                "debugView",
                "OpenPBR Debug View",
                "final",
                {
                    {"Final", "final"},
                    {"Geometry Normal", "geometryNormal"},
                    {"Shading Normal", "shadingNormal"},
                    {"Normal Mapped", "mappedNormal"},
                    {"Tangent", "tangent"},
                    {"Bitangent", "bitangent"},
                    {"Tangent Handedness", "tangentHandedness"},
                    {"Texcoord", "texcoord"},
                    {"Front Face", "frontFace"},
                    {"Material ID", "material"},
                    {"Instance ID", "instance"},
                    {"Triangle ID", "triangle"},
                    {"Base Color", "baseColor"},
                    {"Normal Texture", "normalTexture"},
                    {"Shadow Transmittance", "shadowTransmittance"},
                    {"Shading Side", "shadingSide"},
                },
                true),
            runtimeBoolSetting("debugDisableNormalMap", "Debug: Disable Normal Map", false, true),
            runtimeBoolSetting("debugForceGeometryNormal", "Debug: Force Geometry Normal", false, true),
            runtimeBoolSetting("debugDisableMaterialTextures", "Debug: Disable Material Textures", false, true),
            runtimeBoolSetting("debugDisableDirectLighting", "Debug: Disable Direct Lighting", false, true),
            runtimeBoolSetting("debugUseOpaqueShadows", "Debug: Use Opaque Shadows", false, true),
            runtimeBoolSetting("debugDisableShadows", "Debug: Disable Shadows", false, true),
            runtimeBoolSetting("debugDisableVolumeAttenuation", "Debug: Disable Volume Attenuation", false, true),
            runtimeBoolSetting("debugDisableTransmission", "Debug: Disable Transmission", false, true),
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
    Result<> prepare(const RenderGraphCompileContext& context, std::string& log) override
    {
        // Resource-only graph rebuilds reuse compiled passes. Deferred compile-time
        // settings must rebuild their matching shader and descriptor variants.
        if (visibilityDeferred_ && typedPathTraceProgram_.valid() &&
            (compiledMaterialBinning_ != boolProperty(properties(), "materialBinning", true) ||
                compiledSupplementaryPathTracing_ != boolProperty(properties(), "supplementaryPathTracing", false) ||
                compiledRealtimeDeferred_ != (properties().value("lightingMode", "reference") == "realtime") ||
                compiledExportUpscalerGuides_ != boolProperty(properties(), "exportUpscalerGuides", false))) {
            return compile(context, log);
        }
        return {};
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        streamMaterials_ = context.runtimeScene != nullptr && context.runtimeScene->hasStreamGeometry();
        if (streamMaterials_ && (!visibilityDeferred_ || properties().value("lightingMode", "reference") != "realtime")) {
            log = "StreamAsset scenes require realtime visibility-buffer deferred lighting";
            return makeError(Error::Unsupported);
        }
        const bool supplementaryPathTracing = visibilityDeferred_ &&
            boolProperty(properties(), "supplementaryPathTracing", false);
        streamRayQueries_ = streamMaterials_ && supplementaryPathTracing && std::any_of(context.runtimeScene->materials().begin(),
            context.runtimeScene->materials().end(), [](const auto& material) {
                return material.transmissionFactor > 0.0f || material.alphaMode == "BLEND";
            });
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
        if (!context.preparedScene || !context.preparedScene->snapshot ||
            !context.preparedScene->snapshot->pathTraceResources) {
            log = "Scene resources were not prepared by StreamerSubsystem";
            return makeError(Error::InvalidArgument);
        }
        sceneResources_ = *context.preparedScene->snapshot->pathTraceResources;
        if (!validateMaterialTarget(log)) { return makeError(Error::Unsupported); }
        const auto valuePrograms = sceneResources_.materialBinding()->values();
        std::filesystem::path valueDirectory;
        if (hasValuePrograms() && !valuePrograms->writeInclude(
                PROJECT_SOURCE_DIR "/.cache/materials", valueDirectory, log)) {
            return makeError(Error::Failure);
        }
        const std::string valueSearchPath = valueDirectory.string();
        Result<> result;
        const uint64_t resourceRevision = sceneResources_.revision();
        if (resourceRevision != sceneResourceRevision_) {
            sceneResourceRevision_ = resourceRevision;
            resetAccumulation_ = true;
            hasPreviousCamera_ = false;
        }
        const bool useOpenPBR = useOpenPBRBsdf(properties());
        const bool exportGuides = exportDenoiserGuides(properties());
        const bool ntcActive = sceneResources_.neuralTextures().active();
        const bool positionFetch = !streamMaterials_ && context.device->capabilities().rayTracingPositionFetch;
        const bool ntcCooperativeVector =
            sceneResources_.neuralTextures().cooperativeVectorActive();
        const char* moduleName = nullptr;
        const char* entryPointName = nullptr;
        if (visibilityDeferred_) {
            moduleName = "Features/VisibilityBuffer/VisibilityBufferDeferred";
            entryPointName = materialBinningEnabled(properties())
                ? "visibilityBufferDeferredBinnedMain" : "visibilityBufferDeferredMain";
        } else if (realtime_) {
            moduleName = "Features/Lighting/SceneRealtimeLighting";
            entryPointName = "sceneRealtimeLightingMain";
        } else if (useOpenPBR) {
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
        const uint32_t requestedCacheMode = realtime_ ? kScenePathTraceCacheModeOff : cacheModeFromProperties(properties());
        uint32_t cacheMode = requestedCacheMode;
        std::string cacheWarning;
        if (cacheMode != kScenePathTraceCacheModeOff) {
            if (useOpenPBR || exportGuides) {
                cacheMode = kScenePathTraceCacheModeOff;
                cacheWarning =
                    "ScenePathTracePass radiance cache requires the standard BSDF without denoiser guides; cache disabled\n";
            }
#if METALLIC_HAS_NRC
            else if (cacheMode == kScenePathTraceCacheModeNRC && !context.device->capabilities().rayQuery) {
                cacheMode = kScenePathTraceCacheModeOff;
            }
#else
            else if (cacheMode == kScenePathTraceCacheModeNRC) {
                cacheMode = kScenePathTraceCacheModeOff;
                cacheWarning =
                    "ScenePathTracePass built without the NRC SDK (METALLIC_HAS_NRC=0); NRC cache disabled\n";
            }
#endif
        }
        cacheMode_ = cacheMode;
        const bool standardInlinePathTrace = !useOpenPBR && !exportGuides;
        if (standardInlinePathTrace) { moduleName = "Features/PathTracing/ScenePathTraceInline"; }

        if (!cacheWarning.empty()) {
            log += cacheWarning;
        }

        const bool globalView = visibilityDeferred_ && context.renderView != nullptr &&
            properties().value("sceneBinding", "world") != "asset" && properties().value("viewBinding", "global") != "local";
        const std::string shaderKey = std::string(moduleName) + "." + entryPointName +
            "|valuePrograms=" + std::to_string(hasValuePrograms() ? valuePrograms->key() : 0) +
            "|streamMaterials=" + (streamMaterials_ ? "1" : "0") +
            "|streamRayQueries=" + (streamRayQueries_ ? "1" : "0") +
            "|supplementaryPathTracing=" + (supplementaryPathTracing ? "1" : "0") +
            "|view=" + (globalView ? "1" : "0") +
            "|cache=" + std::to_string(cacheMode_) +
            "|ntc=" + (ntcActive ? "1" : "0") +
            "|coopvec=" + (ntcCooperativeVector ? "1" : "0") +
            "|positionFetch=" + (positionFetch ? "1" : "0") +
            "|lighting=" + properties().value("lightingMode", "reference") +
            "|upscalerGuides=" + (boolProperty(properties(), "exportUpscalerGuides", false) ? "1" : "0");
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

        const bool classified = visibilityDeferred_ && materialBinningEnabled(properties());
        if (classified && (!context.device->capabilities().computeSubgroupBallotArithmetic ||
            context.device->capabilities().subgroupSize != 32)) {
            log += "Deferred material classification requires native wave32; disable materialBinning on this device\n";
            return makeError(Error::Unsupported);
        }
        const bool baseReady = typedPathTraceProgram_.valid() &&
            (!classified || std::all_of(classifiedPrograms_.begin(), classifiedPrograms_.end(),
                [](const ComputeKernel& program) { return program.valid(); }));
        const bool sharcReady = cacheMode_ != kScenePathTraceCacheModeSharc ||
            (sharcTracePrograms_[0].valid() &&
                sharcTracePrograms_[1].valid() &&
                sharcClearProgram_.valid() && sharcResolveProgram_.valid());
        const bool nrcReady = cacheMode_ != kScenePathTraceCacheModeNRC ||
            (nrcTracePrograms_[0].valid() &&
                nrcTracePrograms_[1].valid() &&
                tonemapProgram_.valid());
        if (baseReady && sharcReady && nrcReady) {
            return {};
        }

        if (visibilityDeferred_ && deferredPipelineCache_ == nullptr) {
            result = context.device->createPipelineCache(PipelineCacheDesc{.filePath = PROJECT_SOURCE_DIR "/.cache/pso/VisibilityBufferDeferredPass.pso"}).transform([&](auto rhiValue) { deferredPipelineCache_ = std::move(rhiValue); });
            if (!result || deferredPipelineCache_ == nullptr) {
                log += "createPipelineCache(VisibilityBufferDeferredPass) failed\n";
                return result ? makeError(Error::Failure) : result;
            }
        }

        std::vector<const char*> capabilities{
            "spvRayQueryKHR",
            "spvGroupNonUniformBallot",
        };
        if (positionFetch) {
            capabilities.push_back("spvRayQueryPositionFetchKHR");
        }
        if (ntcCooperativeVector) {
            capabilities.push_back("spvCooperativeVectorNV");
        }

        auto compilePermutation =
            [&](PathTracePermutation permutation,
                std::span<const SlangMacroDefine> extraDefines,
                ComputeKernel& outProgram) -> Result<> {
            std::vector<SlangMacroDefine> defines{
                {.name = "METALLIC_PATH_TRACE_TYPED", .value = "1"},
                {.name = "METALLIC_CUSTOM_MATERIALS", .value = hasValuePrograms() ? "1" : "0"},
                {.name = "METALLIC_STREAM_MATERIALS", .value = streamMaterials_ ? "1" : "0"},
                {.name = "METALLIC_STREAM_RAY_QUERIES", .value = streamRayQueries_ ? "1" : "0"},
                {.name = "METALLIC_GLOBAL_VIEW", .value = globalView ? "1" : "0"},
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
                SlangMacroDefine{
                    .name = "SCENE_RAYQUERY_ENABLE_POSITION_FETCH",
                    .value = positionFetch ? "1" : "0",
                },
            };
            defines.push_back({.name = "METALLIC_DEFERRED_LIGHT_GRID", .value = visibilityDeferred_ ? "1" : "0"});
            defines.push_back({.name = "METALLIC_REALTIME_DEFERRED", .value = visibilityDeferred_ &&
                properties().value("lightingMode", "reference") == "realtime" ? "1" : "0"});
            if (visibilityDeferred_) {
                defines.push_back({.name = "METALLIC_DEFERRED_PATH_TRACING", .value = supplementaryPathTracing ? "1" : "0"});
            }
            defines.push_back({.name = "METALLIC_DEFERRED_UPSCALER_GUIDES", .value = visibilityDeferred_ &&
                boolProperty(properties(), "exportUpscalerGuides", false) ? "1" : "0"});
            defines.insert(defines.end(), extraDefines.begin(), extraDefines.end());
            std::vector<const char*> additionalSearchPaths;
            if (hasValuePrograms()) { additionalSearchPaths.push_back(valueSearchPath.c_str()); }
#if METALLIC_HAS_RTXCR
            additionalSearchPaths.push_back(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
#endif
#if METALLIC_HAS_NTC
            if (ntcActive) {
                additionalSearchPaths.push_back(METALLIC_NTC_SHADER_INCLUDE_DIR);
            }
#endif
            std::shared_ptr<const MaterialExecutableArtifact> artifact;
            const std::string debugName = std::string("ScenePathTracePass.") + toString(permutation);
            std::string diagnostics;
            const bool sharcTrace = permutation == PathTracePermutation::SharcUpdate || permutation == PathTracePermutation::SharcQuery;
            const bool nrcTrace = permutation == PathTracePermutation::NRCUpdate || permutation == PathTracePermutation::NRCQuery;
            const SlangShaderDesc source{.moduleName = sharcTrace ? "Features/PathTracing/ScenePathTraceSharc" : nrcTrace ? "Features/PathTracing/ScenePathTraceNRC" : moduleName, .entryPointName = entryPointName,
                .searchPath = kTriangleShaderSearchPath, .additionalSearchPaths = additionalSearchPaths,
                .capabilities = capabilities, .macroDefines = defines};
            auto compiled = nrcTrace
                ? compileMaterialExecutable(*context.device, source,
                    ComputeKernelDesc{.parameters = parameterAbi<NRCTraceParameters>(kNRCTraceABI, ParameterTransport::InlinePush),
                        .debugName = debugName.c_str()}, nrcTracePrograms_[permutation == PathTracePermutation::NRCUpdate ? 0 : 1], artifact, diagnostics)
                : sharcTrace
                ? compileMaterialExecutable(*context.device, source,
                    ComputeKernelDesc{.parameters = parameterAbi<SharcTraceParameters>(kSharcTraceABI, ParameterTransport::InlinePush),
                        .debugName = debugName.c_str()}, sharcTracePrograms_[permutation == PathTracePermutation::SharcUpdate ? 0 : 1], artifact, diagnostics)
                : compileMaterialExecutable(*context.device, source,
                    ComputeKernelDesc{.parameters = visibilityDeferred_
                        ? parameterAbi<DeferredShadingParameters>(kDeferredShadingABI, ParameterTransport::InlinePush)
                        : realtime_
                        ? parameterAbi<RealtimeLightingParameters>(kRealtimeLightingABI, ParameterTransport::InlinePush)
                        : exportGuides
                        ? parameterAbi<PathTraceGuidesInlineParameters>(kPathTraceGuidesInlineABI, ParameterTransport::InlinePush)
                        : parameterAbi<PathTraceInlineParameters>(kPathTraceInlineABI, ParameterTransport::InlinePush),
                        .debugName = debugName.c_str(), .pipelineCache = deferredPipelineCache_.get()}, outProgram, artifact, diagnostics);
            if (!diagnostics.empty()) { log += diagnostics + '\n'; }
            if (!compiled) {
                // Reload creates replacement passes. Reject the entire transaction
                // so a failure can never replace a previously successful graph.
                if (context.shaderReload || outProgram.valid() || typedPathTraceProgram_.valid()) { return compiled; }
                log += "Initial material compilation failed; displaying the error material.\n";
                std::string errorLog;
                auto fallback = initializeMaterialErrorKernel(*context.device, errorProgram_, errorLog);
                if (!fallback) { log += errorLog; return fallback; }
                return {};
            }
            materialArtifacts_.push_back(std::move(artifact));
            return {};
        };

        if (!typedPathTraceProgram_.valid()) {
            const SlangMacroDefine transmissionClass[] = {{"MATERIAL_CLASS", "4"}};
            result = compilePermutation(
                PathTracePermutation::Base,
                classified ? std::span<const SlangMacroDefine>(transmissionClass) : std::span<const SlangMacroDefine>{},
                typedPathTraceProgram_);
            if (!result) {
                return result;
            }
        }

        if (errorProgram_.valid()) { return {}; }
        if (classified) {
            for (uint32_t type = 0; type < classifiedPrograms_.size(); ++type) {
                if (classifiedPrograms_[type].valid()) { continue; }
                const std::string typeValue = std::to_string(type);
                const SlangMacroDefine classDefine[] = {{"MATERIAL_CLASS", typeValue.c_str()}};
                result = compilePermutation(PathTracePermutation::Base, classDefine, classifiedPrograms_[type]);
                if (!result) { return result; }
            }
        }

        if (errorProgram_.valid()) { return {}; }
        if (cacheMode_ == kScenePathTraceCacheModeSharc) {
            const std::array<SlangMacroDefine, 1> sharcUpdateDefines{
                SlangMacroDefine{.name = "SHARC_UPDATE", .value = "1"},
            };
            if (!sharcTracePrograms_[0].valid()) {
                result = compilePermutation(
                    PathTracePermutation::SharcUpdate,
                    sharcUpdateDefines,
                        sharcTracePrograms_[0]);
                if (!result) {
                    return result;
                }
            }

            const std::array<SlangMacroDefine, 1> sharcQueryDefines{
                SlangMacroDefine{.name = "SHARC_QUERY", .value = "1"},
            };
            if (!sharcTracePrograms_[1].valid()) {
                result = compilePermutation(
                    PathTracePermutation::SharcQuery,
                    sharcQueryDefines,
                        sharcTracePrograms_[1]);
                if (!result) {
                    return result;
                }
            }

            // SHaRC maintenance programs (clear + resolve).
            auto compileMaintenance =
                [&](const char* entryPointName, ComputeKernel& outProgram) -> Result<> {
                ShaderCompileResult maintenanceCompile;
                Result<> maintenanceResult = compileSlangShaderToSpirv(SlangShaderDesc{
                    .moduleName = kSceneSharcMaintenanceShaderModuleName,
                    .entryPointName = entryPointName,
                    .searchPath = kTriangleShaderSearchPath,
                    .capabilities = capabilities,
                }, maintenanceCompile.diagnostics).transform([&](auto value) { maintenanceCompile = std::move(value); });
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
                    ComputeKernelDesc{
                        .spirv = maintenanceCompile.spirv,
                        .parameters = parameterAbi<SharcMaintenanceParams>(kSharcMaintenanceABI, ParameterTransport::InlinePush),
                        .debugName = maintenanceDebugName.c_str(),
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
        if (cacheMode_ == kScenePathTraceCacheModeNRC) {
            const std::array<SlangMacroDefine, 1> nrcUpdateDefines{
                SlangMacroDefine{.name = "NRC_UPDATE", .value = "1"},
            };
            if (!nrcTracePrograms_[0].valid()) {
                result = compilePermutation(
                    PathTracePermutation::NRCUpdate,
                    nrcUpdateDefines,
                        nrcTracePrograms_[0]);
                if (!result) {
                    return result;
                }
            }

            const std::array<SlangMacroDefine, 1> nrcQueryDefines{
                SlangMacroDefine{.name = "NRC_QUERY", .value = "1"},
            };
            if (!nrcTracePrograms_[1].valid()) {
                result = compilePermutation(
                    PathTracePermutation::NRCQuery,
                    nrcQueryDefines,
                        nrcTracePrograms_[1]);
                if (!result) {
                    return result;
                }
            }

            // Tonemap pass producing the final displayable color after the
            // NRC resolve has added the predicted radiance.
            if (!tonemapProgram_.valid()) {
                ShaderCompileResult tonemapCompile;
                Result<> tonemapResult = compileSlangShaderToSpirv(SlangShaderDesc{
                    .moduleName = kScenePathTraceTonemapShaderModuleName,
                    .entryPointName = kScenePathTraceTonemapEntryPointName,
                    .searchPath = kTriangleShaderSearchPath,
                    .capabilities = capabilities,
                }, tonemapCompile.diagnostics).transform([&](auto value) { tonemapCompile = std::move(value); });
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
                    ComputeKernelDesc{
                        .spirv = tonemapCompile.spirv,
                        .parameters = parameterAbi<PathTraceTonemapParams>(kPathTraceTonemapABI, ParameterTransport::InlinePush),
                        .debugName = "ScenePathTracePass.Tonemap",
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
        if (cacheMode_ == kScenePathTraceCacheModeNRC) {
            // Already downgraded to off above; nothing to compile.
        }
#endif

        if (deferredPipelineCache_ != nullptr) {
            const Result<> saveResult = deferredPipelineCache_->save();
            const PipelineCacheStats stats = deferredPipelineCache_->stats();
            spdlog::info("[VisibilityBufferDeferredPass] PSO cache hits={} misses={}",
                stats.hitCount, stats.missCount);
            if (!saveResult) {
                spdlog::warn("[VisibilityBufferDeferredPass] Could not persist PSO cache: {}",
                    resultToString(saveResult));
            }
        }
        compiledShaderKey_ = shaderKey;
        compiledMaterialBinning_ = boolProperty(properties(), "materialBinning", true);
        compiledSupplementaryPathTracing_ = supplementaryPathTracing;
        compiledRealtimeDeferred_ = visibilityDeferred_ && properties().value("lightingMode", "reference") == "realtime";
        compiledExportUpscalerGuides_ = visibilityDeferred_ && boolProperty(properties(), "exportUpscalerGuides", false);
        return {};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        if (errorProgram_.valid()) {
            for (const char* name : {"color", "albedo", "specularAlbedo", "normalRoughness",
                    "motionVectors", "deviceDepth", "linearDepth", "specularHitDistance", "depth"}) {
                auto output = context.outputTexture(name);
                if (!output.valid()) { continue; }
                auto result = dispatchMaterialError(*device_, errorProgram_, context.commandBuffer(),
                    *output.view(), context.width(), context.height(), std::string_view(name) == "color");
                if (!result) { return result; }
            }
            return {};
        }
        CPUProfileRecorder profiler;
        const auto result = executeProfiled(context, visibilityDeferred_ ? &profiler : nullptr);
        context.publishCpuProfile(profiler.sections);
        return result;
    }

    Result<> executeProfiled(RenderGraphExecutionContext& context, CPUProfileRecorder* profiler)
    {
        CPUProfileScope profile(profiler, "Validate scene and environment");
        std::string syncLog;
        // RenderGraph prepares a single scene generation before recording any pass.
        // StreamerSubsystem publishes geometry, material tables and RTAS together;
        // a pass never resolves an authored path midway through this frame.
        if (sceneResources_.revision() != sceneResourceRevision_) {
            if (!validateMaterialTarget(syncLog)) {
                spdlog::error("{}", syncLog);
                return makeError(Error::Unsupported);
            }
            sceneResourceRevision_ = sceneResources_.revision();
            resetAccumulation_ = true;
            hasPreviousCamera_ = false;
            sharcClearPending_ = true;
#if METALLIC_HAS_NRC
            nrcSceneRevision_ = 0;
#endif
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
        const scene::Scene* lightScene = context.runtimeScene();
        if (lightScene == nullptr || context.subsystems() == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        profile.next("Prepare lights and sampling");
        const uint64_t previousLightRevision = lights_.revision();
        const auto resolvedLighting = resolveSceneLighting(lightScene, context.world());
        Result<> lightResult = lights_.update(*device_, context.commandBuffer(), *context.subsystems(),
            lightScene, resolvedLighting);
        if (!lightResult) { return lightResult; }
        if (previousLightRevision != lights_.revision()) {
            resetAccumulation_ = true;
            // Invalidate radiance caches as well as the displayed accumulation.
            sharcClearPending_ = true;
#if METALLIC_HAS_NRC
            nrcSceneRevision_ = 0;
#endif
        }
        if (compiledSupplementaryPathTracing_ || !realtime_ || (visibilityDeferred_ && context.properties().value("lightingMode", "reference") != "realtime")) {
            const auto& bounds = sceneResources_.bounds();
            const float3 center = bounds.valid ? bounds.center() : float3(0.0f);
            ReGIRBuildParameters sampling;
            sampling.lightCount = lights_.lightCount();
            sampling.frameIndex = static_cast<uint32_t>(context.frameIndex());
            sampling.sceneCenter[0] = center.x;
            sampling.sceneCenter[1] = center.y;
            sampling.sceneCenter[2] = center.z;
            sampling.sceneRadius = bounds.valid ? std::max(bounds.radius(), 0.01f) : 1.0f;
            lightResult = lights_.buildSampling(*device_, context.commandBuffer(), *context.subsystems(),
                sampling, 16, 16, true, syncLog);
            if (!lightResult) {
                spdlog::warn("[ScenePathTracePass] Physical light ReGIR build failed: {}", syncLog);
                return lightResult;
            }
        }
        if (environment.resourceRevision != environmentResourceRevision_ ||
            environment.settingsRevision != environmentSettingsRevision_) {
            environmentResourceRevision_ = environment.resourceRevision;
            environmentSettingsRevision_ = environment.settingsRevision;
            resetAccumulation_ = true;
            hasPreviousCamera_ = false;
        }
        TextureHandle color = context.outputTexture("color");
        Buffer* textureFeedback = context.preparedScene() ? context.preparedScene()->textureFeedback : nullptr;
        profile.next("Prepare output and shading parameters");
        const auto& materialTextureViews = sceneResources_.materialTextureViews();
        TextureView* environmentTextureView = environment.radianceView;
        TextureView* environmentImportancePdfView = environment.pdfView;
        const bool useOpenPBR = useOpenPBRBsdf(properties());
        const bool exportGuides = exportDenoiserGuides(properties());
        TextureHandle albedo = exportGuides ? context.outputTexture("albedo") : TextureHandle{};
        TextureHandle specularAlbedo = exportGuides ? context.outputTexture("specularAlbedo") : TextureHandle{};
        TextureHandle normalRoughness = exportGuides ? context.outputTexture("normalRoughness") : TextureHandle{};
        TextureHandle motionVectors = exportGuides ? context.outputTexture("motionVectors") : TextureHandle{};
        TextureHandle linearDepth = exportGuides ? context.outputTexture("linearDepth") : TextureHandle{};
        TextureHandle specularHitDistance = exportGuides ? context.outputTexture("specularHitDistance") : TextureHandle{};
        TextureHandle depth = exportGuides ? context.outputTexture("depth") : TextureHandle{};

        uint32_t cacheMode = cacheMode_;
        if (cacheMode == kScenePathTraceCacheModeSharc) {
            if (!sharcTracePrograms_[1].valid() ||
                !sharcTracePrograms_[0].valid() ||
                !sharcClearProgram_.valid() ||
                !sharcResolveProgram_.valid()) {
                cacheMode = kScenePathTraceCacheModeOff;
            }
        } else if (cacheMode == kScenePathTraceCacheModeNRC) {
#if METALLIC_HAS_NRC
            if (!nrcTracePrograms_[1].valid() ||
                !nrcTracePrograms_[0].valid() ||
                !tonemapProgram_.valid()) {
                cacheMode = kScenePathTraceCacheModeOff;
            }
#else
            cacheMode = kScenePathTraceCacheModeOff;
#endif
        }
        cacheMode_ = cacheMode;


        if (!color.valid() ||
            color.view() == nullptr ||
            !typedPathTraceProgram_.valid() ||
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
                    !validTexture(specularHitDistance) ||
                    !validTexture(depth)))) {
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
        push.outputLinear = 1u;
        TextureView* visibilityView = nullptr;
        TextureView* visibilityDepthView = nullptr;
        TextureView* domainView = nullptr;
        const GPUSceneGlobalBufferViews* deferredViews = nullptr;
        const ClusterLightGridSnapshot* deferredGrid = nullptr;
        const MeshletStreamDeferredGPUResourcesView* deferredStream = nullptr;
        Buffer* deferredFrameInfo = nullptr;
        VisibilityBufferFrameInfo info;
        profile.next("Prepare visibility resources");
        if (visibilityDeferred_) {
            const auto visibility = context.inputTexture("visibility");
            const auto visibilityDepth = context.inputTexture("depth");
            const auto rasterInfo = context.inputBuffer("rasterInfo");
            auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
            if (!visibility.valid() || !visibilityDepth.valid() || !rasterInfo.valid() || gpuScene == nullptr ||
                rasterInfo.desc().size != sizeof(VisibilityBufferFrameInfo) ||
                rasterInfo.desc().memoryLocation != MemoryLocation::HostUpload) {
                return makeError(Error::InvalidArgument);
            }
            const void* mapped = rasterInfo.buffer()->map();
            if (mapped == nullptr) { return makeError(Error::Failure); }
            std::memcpy(&info, mapped, sizeof(info));
            rasterInfo.buffer()->unmap();
            const auto domain = context.inputTexture("domain");
            if ((info.reserved & 1u) != 0u && (!domain.valid() || domain.texture()->desc().format != Format::RGBA32Sfloat)) {
                spdlog::error("Displaced VBuffer requires the matching domain graph input");
                return makeError(Error::InvalidArgument);
            }
            domainView = domain.valid() ? domain.view() : visibilityDepth.view();
            deferredFrameInfo = rasterInfo.buffer();
            if (auto* frame = context.commandBuffer().frameContext()) {
                // rasterInfo is also CPU metadata for downstream passes. Bind an
                // immutable snapshot so the next recording cannot overwrite it
                // while this frame's material classification/decode reads it.
                auto free = std::find_if(deferredFrameInfoPool_.begin(), deferredFrameInfoPool_.end(),
                    [](const auto& buffer) { return buffer.use_count() == 1; });
                if (free == deferredFrameInfoPool_.end()) {
                    std::unique_ptr<Buffer> buffer;
                    auto allocated = device_->createBuffer(rasterInfo.buffer()->desc()).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
                    if (!allocated) { return allocated; }
                    deferredFrameInfoPool_.emplace_back(std::move(buffer));
                    free = std::prev(deferredFrameInfoPool_.end());
                }
                void* destination = (*free)->map();
                if (!destination) { return makeError(Error::Failure); }
                std::memcpy(destination, &info, sizeof(info));
                (*free)->flush({0, sizeof(info)});
                (*free)->unmap();
                deferredFrameInfo = free->get();
                frame->retain(*free);
            }
            deferredStream = gpuScene->visibilityStream({info.lightGridViewIndex, info.lightGridViewGeneration},
                info.frameIndex, info.sceneIdentity);
            if (info.hasStreamGeometry && !deferredStream) { return makeError(Error::InvalidArgument); }
            if (info.width != context.width() || info.height != context.height() || info.frameIndex != context.frameIndex() ||
                lightScene == nullptr || info.sceneIdentity != lightScene->resourceIdentity()) {
                spdlog::error("[VisibilityBufferDeferredPass] Raster scene/view mismatch: raster={}x{} scene={}, deferred={}x{} scene={}",
                    info.width, info.height, info.sceneIdentity, context.width(), context.height(),
                    lightScene != nullptr ? lightScene->resourceIdentity() : 0);
                return makeError(Error::InvalidArgument);
            }
            deferredGrid = gpuScene->lightGrid({info.lightGridViewIndex, info.lightGridViewGeneration}, info.lightGridFrameSlot);
            if (deferredGrid == nullptr || !deferredGrid->valid()) {
                spdlog::error("Deferred lighting requires the current raster view LightGrid");
                return makeError(Error::InvalidArgument);
            }
            deferredViews = &gpuScene->globalBufferViews();
            if (!deferredViews->validFor(gpuScene->drawSet().generation, gpuScene->drawSet().revision)) {
                return makeError(Error::InvalidArgument);
            }
            std::memcpy(push.eye, info.eye, sizeof(info.eye));
            std::memcpy(push.center, info.center, sizeof(info.center));
            std::memcpy(push.upProjection, info.upProjection, sizeof(info.upProjection));
            std::memcpy(push.viewport, info.viewport, sizeof(info.viewport));
            std::memcpy(push.clipOrtho, info.clipOrtho, sizeof(info.clipOrtho));
            // Jitter is metadata, not part of the unjittered camera history.
            push.eye[3] = 0.0f;
            push.center[3] = 0.0f;
            push.samples = uintProperty(context.properties(), "environmentSamples", 64, 1, 256);
            push.deferredSettings = (uintProperty(context.properties(), "transmissionSamples", 2, 1, 16) << 16u) |
                (uintProperty(context.properties(), "transmissionDepth", 8, 2, 16) << 21u);
            visibilityView = visibility.view();
            visibilityDepthView = visibilityDepth.view();
        }
        profile.next("Prepare camera and history");
        push.sampleFrame = static_cast<uint32_t>(context.frameIndex());
        push.temporalJitter = exportGuides ? 1u : 0u;
        if (exportGuides) {
            const std::array<float, 2> jitter = dlssTemporalJitter(context.frameIndex());
            push.jitterOffsetX = jitter[0];
            push.jitterOffsetY = jitter[1];
        }
        if (visibilityDeferred_) {
            push.temporalJitter = info.temporalJitter;
            push.jitterOffsetX = info.jitter[0];
            push.jitterOffsetY = info.jitter[1];
        }
        if (const auto* view = context.viewConstants()) {
            applyViewCamera(view->current, push);
            push.temporalJitter = view->frame[2];
            push.jitterOffsetX = view->jitter[0];
            push.jitterOffsetY = view->jitter[1];
        }
        const ScenePathTraceCameraSnapshot currentCamera = cameraSnapshotFromPush(push);
        const GPUSceneViewId rasterView{info.lightGridViewIndex, info.lightGridViewGeneration};
        if (visibilityDeferred_ && (!hasPreviousCamera_ ||
            std::memcmp(&currentCamera, &previousCamera_, sizeof(currentCamera)) != 0 ||
            deferredHistoryView_ != rasterView ||
            deferredRasterSettingsRevision_ != info.rasterSettingsRevision ||
            deferredHistoryProperties_ != context.properties())) {
            resetAccumulation_ = true;
            deferredHistoryView_ = rasterView;
            deferredRasterSettingsRevision_ = info.rasterSettingsRevision;
            deferredHistoryProperties_ = context.properties();
        }
        const bool previousCameraValid =
            hasPreviousCamera_ &&
            previousCameraWidth_ == context.width() &&
            previousCameraHeight_ == context.height();
        applyPreviousCameraSnapshot(previousCameraValid ? previousCamera_ : currentCamera, push);
        push.previousCameraValid = previousCameraValid ? 1u : 0u;
        if (const auto* view = context.viewConstants()) {
            ScenePathTraceCameraSnapshot previous;
            applyViewCamera(view->previous, previous);
            applyPreviousCameraSnapshot(previous, push);
            push.previousCameraValid = view->frame[1];
        }

        TextureView* historyCurrentView = color.view();
        TextureView* historyPreviousView = color.view();
        Result<> result = prepareHistoryTextures(
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

        profile.next("Prepare material textures and LUTs");
        if (useOpenPBR) {
            result = openPBRLuts_.upload(context.commandBuffer());
            if (!result) {
                return result;
            }
        }

        const auto encodePathTraceResources = [&](ParameterWriter& writer) {
            writer.retain(std::make_shared<ScenePathTraceResources>(sceneResources_));
            PathTraceParameters params{};
            params.settings = writer.data(&push, sizeof(push));
            params.output = writer.storageImage(color.view());
            if (!streamMaterials_ || streamRayQueries_) {
                auto* acceleration = context.inputAccelerationStructure("accelerationStructure");
                if (!acceleration) { acceleration = streamMaterials_ ? deferredStream->accelerationStructure
                    : sceneResources_.accelerationStructure().accelerationStructure(); }
                params.scene = writer.accelerationStructure(acceleration);
            }
            if (!streamMaterials_) {
                params.vertices = writer.dataBuffer(sceneResources_.shadingVertexBuffer(), 16, 8);
                params.indices = writer.dataBuffer(sceneResources_.indexBuffer(), 4, 4);
                params.primitives = writer.dataBuffer(sceneResources_.primitiveBuffer(), 32, 4);
                params.instances = writer.dataBuffer(sceneResources_.instanceBuffer(), 16, 4);
                if (sceneResources_.fallbackPositionBuffer()) {
                    params.positions = writer.dataBuffer(sceneResources_.fallbackPositionBuffer(), 12, 4);
                }
            }
            params.materials = writer.buffer(sceneResources_.materialBuffer());
            params.historyCurrent = writer.storageImage(historyCurrentView);
            params.historyPrevious = writer.storageImage(historyPreviousView);
            params.materialTextures = writer.sampledImages(materialTextureViews);
            params.environment = writer.sampledImage(environmentTextureView);
            params.environmentPdf = writer.sampledImage(environmentImportancePdfView);
            if (useOpenPBR) {
                params.lut2D = writer.sampledImages(openPBRLuts_.lut2DViews());
                params.lut3D = writer.sampledImages(openPBRLuts_.lut3DViews());
            }
            params.lights = writer.buffer(lights_.buffer());
            if (compiledSupplementaryPathTracing_ || !realtime_ ||
                (visibilityDeferred_ && properties().value("lightingMode", "reference") != "realtime")) {
                params.reGIR = writer.buffer(lights_.reGIRBuffer());
                params.punctualPdf = writer.sampledImage(lights_.lightPdfView());
            }
            if (exportGuides) {
                params.albedo = writer.storageImage(albedo.view());
                params.specularAlbedo = writer.storageImage(specularAlbedo.view());
                params.normalRoughness = writer.storageImage(normalRoughness.view());
                params.motionVectors = writer.storageImage(motionVectors.view());
                params.linearDepth = writer.storageImage(linearDepth.view());
                params.specularHitDistance = writer.storageImage(specularHitDistance.view());
                params.depth = writer.storageImage(depth.view());
            }
            if (hasValuePrograms()) { params.materialValues = writer.buffer(sceneResources_.materialBinding()->valueBuffer()); }
            const auto& neural = sceneResources_.neuralTextures();
            if (neural.active()) {
                params.ntcLatents = writer.sampledImages(neural.latentTextureViews());
                params.ntcConstants = writer.buffer(neural.constantsBuffer());
                params.ntcWeights = writer.buffer(neural.weightsBuffer());
                params.ntcInfo = writer.buffer(neural.setInfoBuffer());
                params.ntcSampler = writer.sampler(neural.latentSampler());
            }
            return params;
        };

        profile.next("Prepare deferred stages");
        MaterialBinningResult materialBins;
        ScreenSpaceShadowResult shadow;
        if (visibilityDeferred_) {
            if (context.properties().value("lightingMode", "reference") == "realtime") {
                const auto externalShadow = context.inputTexture("shadow");
                const auto externalParameters = context.inputBuffer("shadowParameters");
                if (externalShadow.valid() != externalParameters.valid()) {
                    spdlog::error("Deferred shadows require both shadow and shadowParameters inputs");
                    return makeError(Error::InvalidArgument);
                }
                if (externalShadow.valid()) {
                    if (externalShadow.desc().width != context.width() || externalShadow.desc().height != context.height() ||
                        externalParameters.desc().size != sizeof(ScreenSpaceShadowParameters)) {
                        return makeError(Error::InvalidArgument);
                    }
                    shadow = {.texture = externalShadow.texture(), .shadow = externalShadow.view(),
                        .parameters = externalParameters.buffer()};
                } else {
                    CPUProfileScope shadowProfile(profiler, "Record inline shadows");
                    // Preserve realtime graphs authored before the explicit shadow stage.
                    if (context.streamer() == nullptr) { return makeError(Error::InvalidArgument); }
                    ViewConstants shadowView{};
                    if (const auto* view = context.viewConstants()) {
                        shadowView = *view;
                    } else {
                        std::memcpy(&shadowView.current, push.eye, sizeof(ViewCameraConstants));
                        std::memcpy(&shadowView.previous, push.previousEye, sizeof(ViewCameraConstants));
                        shadowView.jitter[0] = push.jitterOffsetX;
                        shadowView.jitter[1] = push.jitterOffsetY;
                        shadowView.jitter[2] = previousShadowJitter_[0];
                        shadowView.jitter[3] = previousShadowJitter_[1];
                        shadowView.frame[0] = push.sampleFrame;
                        shadowView.frame[1] = push.previousCameraValid;
                        shadowView.frame[2] = push.temporalJitter;
                    }
                    const auto settings = screenSpaceShadowSettings(context.properties());
                    const auto lightRecords = buildScreenSpaceShadowLightRecords(lightScene, resolvedLighting);
                    std::string shadowLog;
                    result = shadows_.record(*device_, context.commandBuffer(), *context.streamer(), *visibilityDepthView, shadowView, lightRecords, sceneResources_.revision(), lightScene->transformRevision(), settings, shadowLog, &sceneResources_, streamMaterials_ ? deferredStream : nullptr, profiler, context.inputAccelerationStructure("accelerationStructure")).transform([&](auto value) { shadow = std::move(value); });
                    if (!result) { spdlog::error("Ray-traced shadows: {} ({})", shadowLog, resultToString(result)); return result; }
                    previousShadowJitter_ = {shadowView.jitter[0], shadowView.jitter[1]};
                }
            }
            if (streamRayQueries_) {
                if (deferredStream == nullptr || deferredStream->accelerationStructure == nullptr) {
                    spdlog::error("Stream BLEND/transmission requires enableClusterRtx=true and a ready stream TLAS");
                    return makeError(Error::InvalidArgument);
                }
                auto registry = device_->resourceRegistry();
                if (!registry) { return makeError(registry.error()); }
                ParameterWriter writer(*device_, **registry, context.commandBuffer().frameContext());
                auto encoded = deferredStream->encodeRayQueryParameters(writer);
                if (!encoded) { return makeError(encoded.error()); }
                result = encoded->bindResources(context.commandBuffer());
                if (!result) { return result; }
                push.streamScene = encoded->address();
            }
            Buffer* fallback = deferredViews->geometries.buffer;
            if (materialBinningEnabled(context.properties())) {
                CPUProfileScope binningProfile(profiler, "Record material binning");
                std::string binningLog;
                result = materialBinning_.record(*device_, context.commandBuffer(), {
                    .visibility = visibilityView, .records = deferredViews->meshletDraws.buffer
                        ? deferredViews->meshletDraws.buffer : fallback,
                    .instances = deferredViews->instances.buffer, .materials = deferredViews->materials.buffer,
                    .shadingMaterials = sceneResources_.materialBuffer(),
                    .width = push.width, .height = push.height,
                    .streamRecords = deferredStream ? deferredStream->visibleClusterBuffer : nullptr,
                    .streamGroups = deferredStream ? deferredStream->activeGroupBuffer : nullptr,
                    .residentRecordCount = info.residentRecordCount}, binningLog).transform([&](auto value) { materialBins = std::move(value); });
                if (!result) { spdlog::error("Material binning: {}", binningLog); return result; }
            }
        }

        profile.next("Record shading dispatch");
        if (cacheMode == kScenePathTraceCacheModeSharc) {
            result = executeSharcFrame(
                context,
                push,
                encodePathTraceResources,
                sharcTracePrograms_[1],
                sharcTracePrograms_[0]);
            if (!result) {
                return result;
            }
        } else if (cacheMode == kScenePathTraceCacheModeNRC) {
#if METALLIC_HAS_NRC
            result = executeNrcFrame(
                context,
                push,
                encodePathTraceResources,
                nrcTracePrograms_[0],
                nrcTracePrograms_[1],
                historyCurrentView,
                historyPreviousView);
            if (!result) {
                return result;
            }
#endif
        } else {
            auto resources = stageResources(context, push);
            if (materialBins.arguments != nullptr) {
                const std::array buffers{materialBins.bins, materialBins.tiles, materialBins.arguments};
                const std::array names{"materialBins", "materialTiles", "materialArguments"};
                for (size_t i = 0; i < buffers.size(); ++i) {
                    result = importBuffer(resources, names[i], buffers[i]);
                    if (!result) { return result; }
                    resources.uses.push_back({names[i], i == 2 ? RenderGraphResourceAccess::BufferIndirectRead
                        : RenderGraphResourceAccess::BufferShaderRead});
                }
            }
            const std::array stages{RenderGraphStage{visibilityDeferred_ ? "Deferred shading" : "Path trace shading", resources.uses,
                [&](CommandBuffer& commands) -> Result<> {
                    if (typedPathTraceProgram_.valid()) {
                        auto registry = device_->resourceRegistry();
                        if (!registry) { return makeError(registry.error()); }
                        ParameterWriter writer(*device_, **registry, commands.frameContext());
                        PathTraceParameters params = encodePathTraceResources(writer);
                        Result<EncodedParameters> encoded;
                        PathTraceInlineParameters root{};
                        root.settings = params.settings;
                        root.output = params.output;
                        root.historyCurrent = params.historyCurrent;
                        root.historyPrevious = params.historyPrevious;
                        params.settings = 0;
                        params.output = {};
                        params.historyCurrent = {};
                        params.historyPrevious = {};
                        if (visibilityDeferred_) {
                            DeferredShadingResources resources{};
                            const GPUSceneBufferView* views[] = {&deferredViews->vertices, &deferredViews->meshlets,
                                &deferredViews->meshletDraws, &deferredViews->meshletVertices, &deferredViews->meshletTriangleWords,
                                &deferredViews->geometries, &deferredViews->instances, &deferredViews->materials};
                            ShaderDataSpan* spans[] = {&resources.vertices, &resources.meshlets, &resources.records,
                                &resources.meshletVertices, &resources.triangles, &resources.geometries,
                                &resources.instances, &resources.materials};
                            for (size_t i = 0; i < std::size(views); ++i) {
                                const auto& view = *views[i];
                                if (!view.buffer) { continue; }
                                auto slice = view.buffer->slice({view.offset, view.size});
                                if (!slice) { return makeError(slice.error()); }
                                *spans[i] = writer.dataBuffer(*slice, view.structureStride, 4);
                            }
                            resources.irradiance = writer.buffer(environment.sphericalHarmonicsBuffer);
                            resources.gridParams = writer.buffer(deferredGrid->parameters);
                            resources.gridLights = writer.buffer(deferredGrid->lights);
                            resources.gridCandidates = writer.buffer(deferredGrid->candidates);
                            resources.gridCells = writer.buffer(deferredGrid->cells);
                            resources.gridIndices = writer.buffer(deferredGrid->lightIndices);
                            if (context.viewConstantsBuffer()) { resources.view = writer.buffer(context.viewConstantsBuffer()); }
                            if (shadow.shadow) {
                                resources.shadow = writer.sampledImage(shadow.shadow);
                                resources.shadowParams = writer.buffer(shadow.parameters);
                                resources.specular = writer.buffer(environment.prefilteredSpecularBuffer);
                            }
                            resources.frameInfo = writer.buffer(deferredFrameInfo);
                            resources.feedback = writer.buffer(textureFeedback);
                            resources.sampler = writer.sampler(materialSampler_);
                            if (deferredStream) {
                                resources.streamRecords = writer.buffer(deferredStream->visibleClusterBuffer);
                                resources.streamGroups = writer.buffer(deferredStream->activeGroupBuffer);
                                resources.streamPages = writer.buffer(deferredStream->pageBuffer);
                                resources.streamTable = writer.buffer(deferredStream->pageTableBuffer);
                                resources.streamParams = writer.buffer(deferredStream->paramsBuffer);
                            }
                            if (materialBins.arguments) {
                                resources.bins = writer.dataBuffer(materialBins.bins, 8, 8);
                                resources.tiles = writer.dataBuffer(materialBins.tiles, 8, 8);
                            }
                            DeferredShadingParameters deferred{};
                            root.resources = writer.data(&params, sizeof(params));
                            deferred.path = root;
                            deferred.resources = writer.data(&resources, sizeof(resources));
                            deferred.visibility = writer.sampledImage(visibilityView);
                            deferred.depth = writer.sampledImage(visibilityDepthView);
                            deferred.domain = writer.sampledImage(domainView);
                            if (boolProperty(context.properties(), "exportUpscalerGuides", false)) {
                                deferred.motion = writer.storageImage(context.outputTexture("motionVectors").view());
                                deferred.deviceDepth = writer.storageImage(context.outputTexture("deviceDepth").view());
                            }
                            if (materialBins.arguments) {
                                if (materialBins.binCount > kMaterialClassCount) { return makeError(Error::InvalidArgument); }
                                std::vector<ComputeIndirectParameters> dispatches;
                                dispatches.reserve(materialBins.binCount);
                                for (uint32_t bin = 0; bin < materialBins.binCount; ++bin) {
                                    deferred.binIndex = bin;
                                    auto packet = writer.encode(deferred, kDeferredShadingABI, ParameterTransport::InlinePush);
                                    if (!packet) { return makeError(packet.error()); }
                                    auto arguments = materialBins.arguments->slice({uint64_t(bin) * 12, 12});
                                    if (!arguments) { return makeError(arguments.error()); }
                                    dispatches.push_back({.parameters = std::move(*packet), .arguments = *arguments,
                                        .kernel = bin < classifiedPrograms_.size() ? &classifiedPrograms_[bin] : &typedPathTraceProgram_});
                                }
                                auto prepared = typedPathTraceProgram_.prepareIndirectBatch(dispatches);
                                if (!prepared) { return makeError(prepared.error()); }
                                return prepared->record(commands);
                            }
                            encoded = writer.encode(deferred, kDeferredShadingABI, ParameterTransport::InlinePush);
                        } else if (realtime_) {
                            root.resources = writer.data(&params, sizeof(params));
                            RealtimeLightingParameters realtime{};
                            realtime.path = root;
                            realtime.irradiance = writer.buffer(environment.sphericalHarmonicsBuffer);
                            encoded = writer.encode(realtime, kRealtimeLightingABI, ParameterTransport::InlinePush);
                        } else if (exportGuides) {
                            PathTraceGuidesInlineParameters guides{};
                            guides.albedo = params.albedo;
                            guides.specularAlbedo = params.specularAlbedo;
                            guides.normalRoughness = params.normalRoughness;
                            guides.motionVectors = params.motionVectors;
                            guides.linearDepth = params.linearDepth;
                            guides.specularHitDistance = params.specularHitDistance;
                            guides.depth = params.depth;
                            params.albedo = {};
                            params.specularAlbedo = {};
                            params.normalRoughness = {};
                            params.motionVectors = {};
                            params.linearDepth = {};
                            params.specularHitDistance = {};
                            params.depth = {};
                            root.resources = writer.data(&params, sizeof(params));
                            guides.path = root;
                            encoded = writer.encode(guides, kPathTraceGuidesInlineABI, ParameterTransport::InlinePush);
                        } else {
                            root.resources = writer.data(&params, sizeof(params));
                            encoded = writer.encode(root, kPathTraceInlineABI, ParameterTransport::InlinePush);
                        }
                        if (!encoded) { return makeError(encoded.error()); }
                        return typedPathTraceProgram_.dispatch(commands, *encoded,
                            (context.width() + 7) / 8, (context.height() + 7) / 8);
                    }
                    return makeError(Error::InvalidArgument);
                }}};
            result = context.executeStages(stages, resources.buffers, resources.textures);
            if (!result) { return result; }
        }

        profile.next("Publish shading history");
        if (push.enableAccumulation != 0 && context.historyResources() != nullptr) {
            const auto name = historyNameForContext(context, push.cacheMode);
            result = context.historyResources()->publishTextureState(context.commandBuffer(), name,
                HistorySlot::Current, ResourceState::General, true);
            if (result) { result = context.historyResources()->publishTextureState(context.commandBuffer(), name,
                HistorySlot::Previous, ResourceState::General); }
            if (!result) { return result; }
        }
        previousCamera_ = currentCamera;
        previousCameraWidth_ = context.width();
        previousCameraHeight_ = context.height();
        hasPreviousCamera_ = true;
        return {};
    }

private:
    std::vector<std::shared_ptr<Buffer>> deferredFrameInfoPool_;

    struct StageResources {
        std::vector<RenderGraphStageUse> uses;
        std::vector<RenderGraphTextureImport> textures;
        std::vector<RenderGraphBufferImport> buffers;
    };

    StageResources stageResources(RenderGraphExecutionContext& context, const ScenePathTracePush& push) const
    {
        using Access = RenderGraphResourceAccess;
        StageResources resources;
        resources.uses.push_back({"output.color", Access::TextureStorageReadWrite});
        if (visibilityDeferred_) {
            for (const char* name : {"visibility", "depth", "domain", "shadow"}) {
                if (context.inputTexture(name).valid()) { resources.uses.push_back({name, Access::TextureSampleRead}); }
            }
            if (context.inputBuffer("shadowParameters").valid()) {
                resources.uses.push_back({"shadowParameters", Access::BufferShaderRead});
            }
            if (boolProperty(context.properties(), "exportUpscalerGuides", false)) {
                resources.uses.push_back({"output.motionVectors", Access::TextureStorageWrite});
                resources.uses.push_back({"deviceDepth", Access::TextureStorageWrite});
            }
        }
        if (exportDenoiserGuides(context.properties())) {
            for (const char* name : {"albedo", "specularAlbedo", "normalRoughness", "motionVectors",
                "linearDepth", "specularHitDistance", "depth"}) {
                resources.uses.push_back({name, Access::TextureStorageWrite});
            }
        }
        if (push.enableAccumulation != 0) {
            const auto name = historyNameForContext(context, push.cacheMode);
            const auto current = context.historyResources()->texture(name, HistorySlot::Current);
            const auto previous = context.historyResources()->texture(name, HistorySlot::Previous);
            resources.textures.push_back({"historyCurrent", current.texture, current.view, current.state, ResourceState::General});
            resources.textures.push_back({"historyPrevious", previous.texture, previous.view, previous.state, ResourceState::General});
            resources.uses.push_back({"historyCurrent", Access::TextureStorageReadWrite});
            resources.uses.push_back({"historyPrevious", Access::TextureStorageRead});
        }
        // With accumulation disabled both shader history bindings alias color.
        // Its graph declaration already covers them; a private import would
        // incorrectly grant a second ownership contract to the same allocation.
        return resources;
    }

    static Result<> importBuffer(StageResources& resources, std::string_view name, Buffer* buffer)
    {
        if (!buffer) { return makeError(Error::InvalidArgument); }
        auto slice = buffer->slice({0, buffer->desc().size});
        if (!slice) { return makeError(slice.error()); }
        resources.buffers.push_back({name, *slice});
        return {};
    }

    static bool validTexture(TextureHandle texture)
    {
        return texture.valid() && texture.texture() != nullptr && texture.view() != nullptr;
    }

    bool exportDenoiserGuides(const RenderGraphProperties& properties) const
    {
        return !realtime_ && boolProperty(properties, "exportDenoiserGuides", false);
    }

    Result<> prepareHistoryTextures(
        RenderGraphExecutionContext& context,
        TextureView& fallbackView,
        ScenePathTracePush& push,
        TextureView*& outCurrentView,
        TextureView*& outPreviousView)
    {
        HistoryResourceManager* history = context.historyResources();
        // NRC mode always routes through the linear HDR history texture: the
        // resolve pass adds predicted radiance before a separate tonemap pass.
        const bool debugViewEnabled =
            useOpenPBRBsdf(context.properties()) && push.debugView != kScenePathTraceDebugViewFinal;
        const bool accumulationEnabled = (!realtime_ || visibilityDeferred_) && !debugViewEnabled &&
            !(visibilityDeferred_ && (context.properties().value("lightingMode", "reference") == "realtime" ||
                boolProperty(context.properties(), "exportUpscalerGuides", false))) &&
            (push.cacheMode == kScenePathTraceCacheModeNRC ||
                boolProperty(context.properties(), "accumulate", true));
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

        const bool nrcHistory = push.cacheMode == kScenePathTraceCacheModeNRC;
        // NRC's native resolve shader declares rgba32f storage output.
        const Format historyFormat = nrcHistory ? Format::RGBA32Sfloat :
            exportDenoiserGuides(context.properties()) ? Format::RGBA16Sfloat :
            Format::RGBA32Sfloat;
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
        const std::string historyName = historyNameForContext(context, push.cacheMode);
        Result<> result = history->ensureTexture(
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

        outCurrentView = current.view;
        outPreviousView = previous.view;

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
            return kScenePathTraceCacheModeNRC;
        }
        return kScenePathTraceCacheModeOff;
    }

    void clearPrograms()
    {
        errorProgram_.clear();
        typedPathTraceProgram_.clear();
        for (auto& program : sharcTracePrograms_) { program.clear(); }
        for (auto& program : nrcTracePrograms_) { program.clear(); }
        materialArtifacts_.clear();
        for (ComputeKernel& program : classifiedPrograms_) { program.clear(); }
        sharcClearProgram_.clear();
        sharcResolveProgram_.clear();
        tonemapProgram_.clear();
        materialBinning_.clear();
    }

    Result<> ensureSharcBuffers(Device& device, uint32_t entryCount)
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
            Result<> result = device.createBuffer(bufferDesc).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
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

    Result<> executeSharcFrame(
        RenderGraphExecutionContext& context,
        ScenePathTracePush& push,
        const std::function<PathTraceParameters(ParameterWriter&)>& encodeResources,
        ComputeKernel& queryProgram,
        ComputeKernel& updateProgram)
    {
        const uint32_t entriesLog2 = uintProperty(
            context.properties(),
            "sharc.entriesLog2",
            kSharcDefaultEntriesLog2,
            kSharcMinEntriesLog2,
            kSharcMaxEntriesLog2);
        const uint32_t entryCount = 1u << entriesLog2;

        Result<> result = ensureSharcBuffers(*device_, entryCount);
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

        params.sharcAccumulationFrameNum = uintProperty(
            context.properties(),
            "sharc.maxAccumulatedFrames",
            kSharcDefaultMaxAccumulatedFrames,
            1,
            1024);
        params.sharcStaleFrameNumMax = uintProperty(
            context.properties(),
            "sharc.staleFrameNum",
            kSharcDefaultStaleFrameNum,
            8,
            1024);

        // One immutable named-resource packet is shared by update and query.
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, commandBuffer.frameContext());
        auto scene = encodeResources(writer);
        SharcTraceParameters trace{};
        trace.path.settings = scene.settings;
        trace.path.output = scene.output;
        trace.path.historyCurrent = scene.historyCurrent;
        trace.path.historyPrevious = scene.historyPrevious;
        scene.settings = 0;
        scene.output = {};
        scene.historyCurrent = {};
        scene.historyPrevious = {};
        trace.path.resources = writer.data(&scene, sizeof(scene));
        trace.cacheSettings = writer.data(&params, sizeof(params));
        trace.hashEntries = writer.buffer(sharcHashEntriesBuffer_.get());
        trace.accumulation = writer.buffer(sharcAccumulationBuffer_.get());
        trace.resolved = writer.buffer(sharcResolvedBuffer_.get());
        auto encoded = writer.encode(trace, kSharcTraceABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        const uint32_t stride = std::max(params.sharcUpdateStride, 1u);
        const uint32_t updateWidth = (push.width + stride - 1u) / stride;
        const uint32_t updateHeight = (push.height + stride - 1u) / stride;
        // SHaRC resolve: combine per-frame accumulation with previous data.
        SceneSharcMaintenancePush resolvePush{};
        copyFloat4(push.eye, resolvePush.cameraPosition);
        copyFloat4(push.previousEye, resolvePush.cameraPositionPrev);
        resolvePush.sceneScale = params.sharcSceneScale;
        resolvePush.entriesNum = sharcEntryCount_;
        resolvePush.accumulationFrameNum = params.sharcAccumulationFrameNum;
        resolvePush.staleFrameNumMax = params.sharcStaleFrameNumMax;
        resolvePush.frameIndex = push.accumulationFrame;
        // SHaRC render/query at full resolution with early termination.
        auto resources = stageResources(context, push);
        const std::array buffers{sharcHashEntriesBuffer_.get(), sharcAccumulationBuffer_.get(), sharcResolvedBuffer_.get()};
        const std::array names{"sharcHash", "sharcAccumulation", "sharcResolved"};
        std::vector<RenderGraphStageUse> maintenanceUses;
        for (size_t i = 0; i < buffers.size(); ++i) {
            result = importBuffer(resources, names[i], buffers[i]);
            if (!result) { return result; }
            const auto access = RenderGraphResourceAccess::BufferStorageReadWrite;
            maintenanceUses.push_back({names[i], access});
            resources.uses.push_back({names[i], access});
        }
        SceneSharcMaintenancePush clearPush{};
        clearPush.entriesNum = sharcEntryCount_;
        const uint32_t maintenanceGroups = (sharcEntryCount_ + kSharcMaintenanceBlockSize - 1u) / kSharcMaintenanceBlockSize;
        std::vector<RenderGraphStage> stages;
        const bool clearCache = sharcClearPending_ || *sharcDiscarded_;
        if (clearCache) {
            stages.push_back({"SHaRC clear", maintenanceUses, [&](CommandBuffer& commands) {
                return dispatchSharcMaintenance(commands, sharcClearProgram_, clearPush, maintenanceGroups);
            }});
        }
        stages.push_back({"SHaRC update", resources.uses, [&](CommandBuffer& commands) {
            return updateProgram.dispatch(commands, *encoded, (updateWidth + 7) / 8, (updateHeight + 7) / 8);
        }});
        stages.push_back({"SHaRC resolve", maintenanceUses, [&](CommandBuffer& commands) {
            return dispatchSharcMaintenance(commands, sharcResolveProgram_, resolvePush, maintenanceGroups);
        }});
        stages.push_back({"SHaRC query", resources.uses, [&](CommandBuffer& commands) {
            return queryProgram.dispatch(commands, *encoded, (push.width + 7) / 8, (push.height + 7) / 8);
        }});
        const auto discarded = sharcDiscarded_;
        result = commandBuffer.addSubmissionTransaction(std::make_shared<SubmissionTransaction>([] {},
            [discarded] { *discarded = true; }));
        if (!result) { return result; }
        result = context.executeStages(stages, resources.buffers, resources.textures);
        if (result) { sharcClearPending_ = false; *sharcDiscarded_ = false; }
        return result;
    }

    Result<> dispatchSharcMaintenance(
        CommandBuffer& commandBuffer,
        ComputeKernel& program,
        const SceneSharcMaintenancePush& maintenancePush,
        uint32_t groupCount)
    {
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, commandBuffer.frameContext());
        const SharcMaintenanceParams params{
            .hashEntries = writer.buffer(sharcHashEntriesBuffer_.get()),
            .accumulation = writer.buffer(sharcAccumulationBuffer_.get()),
            .resolved = writer.buffer(sharcResolvedBuffer_.get()),
            .settings = maintenancePush,
        };
        auto encoded = writer.encode(params, kSharcMaintenanceABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return program.dispatch(commandBuffer, *encoded, std::max(groupCount, 1u));
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

    Result<> executeNrcFrame(
        RenderGraphExecutionContext& context,
        ScenePathTracePush& push,
        const std::function<PathTraceParameters(ParameterWriter&)>& encodeResources,
        ComputeKernel& updateProgram,
        ComputeKernel& queryProgram,
        TextureView* historyCurrentView,
        TextureView* historyPreviousView)
    {
        if (device_ == nullptr || graphicsQueue_ == nullptr ||
            historyCurrentView == nullptr || historyPreviousView == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        CommandBuffer& commandBuffer = context.commandBuffer();

        // NRC requires EndFrame after submission; defer it to the next frame's
        // execute, when the previous command buffer is guaranteed submitted.
        if (*nrcEndFramePending_) {
            *nrcEndFramePending_ = false;
            Result<> endResult = nrc_.endFrame(*graphicsQueue_);
            if (!endResult) {
                return endResult;
            }
        }
        if (!nrc_.valid()) {
            std::string nrcLog;
            Result<> initResult = nrc_.initialize(*device_, nrcLog);
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
            *nrcDiscarded_ ||
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
        settings.maxPathVertices = kNRCMaxPathVertices;

        const bool reconfigure =
            !nrcConfigured_ || settings != nrcContextSettings_ || settings.requestReset;
        if (reconfigure) {
            std::string configureLog;
            Result<> configureResult = nrc_.configure(settings, *device_, configureLog);
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
        // BeginFrame supplies constants; encode their immutable snapshot only
        // after it succeeds, before either trace stage records a dispatch.
        ScenePathTraceCacheParams params;
        Result<> result;
        EncodedParameters traceEncoded;
        auto prepareParams = [&]() -> Result<> {
            const auto constants = nrc_.populateShaderConstants();
            if (!constants) {
                return makeError(constants.error());
            }
            const auto& nrcConstants = *constants;

            copyFloat4(push.eye, params.sharcCameraPosition);
            copyFloat4(push.previousEye, params.sharcCameraPositionPrev);
            params.sharcEntriesNum = 0;
            params.frameIndex = push.accumulationFrame;
            params.cacheMode = kScenePathTraceCacheModeNRC;
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
            auto registry = device_->resourceRegistry();
            if (!registry) { return makeError(registry.error()); }
            ParameterWriter writer(*device_, **registry, commandBuffer.frameContext());
            auto scene = encodeResources(writer);
            NRCTraceParameters trace{};
            trace.path.settings = scene.settings;
            trace.path.output = scene.output;
            trace.path.historyCurrent = scene.historyCurrent;
            trace.path.historyPrevious = scene.historyPrevious;
            scene.settings = 0;
            scene.output = {};
            scene.historyCurrent = {};
            scene.historyPrevious = {};
            trace.path.resources = writer.data(&scene, sizeof(scene));
            trace.cacheSettings = writer.data(&params, sizeof(params));
            trace.queryPath = writer.buffer(nrc_.buffer(static_cast<uint32_t>(nrc::BufferIdx::QueryPathInfo)));
            trace.trainingPath = writer.buffer(nrc_.buffer(static_cast<uint32_t>(nrc::BufferIdx::TrainingPathInfo)));
            trace.vertices = writer.buffer(nrc_.buffer(static_cast<uint32_t>(nrc::BufferIdx::TrainingPathVertices)));
            trace.radiance = writer.buffer(nrc_.buffer(static_cast<uint32_t>(nrc::BufferIdx::QueryRadianceParams)));
            trace.counters = writer.buffer(nrc_.buffer(static_cast<uint32_t>(nrc::BufferIdx::Counter)));
            auto encoded = writer.encode(trace, kNRCTraceABI, ParameterTransport::InlinePush);
            if (!encoded) { return makeError(encoded.error()); }
            traceEncoded = std::move(*encoded);
            return {};

        };

        static constexpr nrc::BufferIdx kTraceBuffers[] = {
            nrc::BufferIdx::QueryPathInfo,
            nrc::BufferIdx::TrainingPathInfo,
            nrc::BufferIdx::TrainingPathVertices,
            nrc::BufferIdx::QueryRadianceParams,
            nrc::BufferIdx::Counter,
        };
        const uint32_t trainingWidth = std::max(nrcContextSettings_.trainingDimensions.x, 1u);
        const uint32_t trainingHeight = std::max(nrcContextSettings_.trainingDimensions.y, 1u);
        ScenePathTraceTonemapPush tonemapPush{
            .width = push.width,
            .height = push.height,
            .exposure = context.world() != nullptr ? std::exp2(-context.world()->lighting().exposureEV100) : 1.0f,
            .outputLinear = 1u,
            .hasHistory = push.hasHistory,
            .accumulationFrame = push.accumulationFrame,
        };
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter tonemapWriter(*device_, **registry, commandBuffer.frameContext());
        const PathTraceTonemapParams tonemapParams{
            .source = tonemapWriter.storageImage(historyCurrentView),
            .output = tonemapWriter.storageImage(context.outputTexture("color").view()),
            .historyPrevious = tonemapWriter.storageImage(historyPreviousView),
            .settings = tonemapPush,
        };
        auto tonemapEncoded = tonemapWriter.encode(tonemapParams, kPathTraceTonemapABI, ParameterTransport::InlinePush);
        if (!tonemapEncoded) { return makeError(tonemapEncoded.error()); }
        using Access = RenderGraphResourceAccess;
        auto resources = stageResources(context, push);
        std::array<std::string, vulkan::NRCIntegration::kBufferCount> names;
        std::vector<RenderGraphStageUse> sdkUses;
        for (uint32_t i = 0; i < names.size(); ++i) {
            auto* buffer = nrc_.buffer(i);
            if (!buffer) { continue; }
            names[i] = "nrcBuffer" + std::to_string(i);
            result = importBuffer(resources, names[i], buffer);
            if (!result) { return result; }
            sdkUses.push_back({names[i], Access::BufferStorageReadWrite});
        }
        for (const auto buffer : kTraceBuffers) {
            const auto i = static_cast<size_t>(buffer);
            if (names[i].empty()) { return makeError(Error::InvalidArgument); }
            resources.uses.push_back({names[i], Access::BufferStorageReadWrite});
        }
        auto resolveUses = sdkUses;
        const char* currentHistory = push.enableAccumulation ? "historyCurrent" : "output.color";
        const char* previousHistory = push.enableAccumulation ? "historyPrevious" : "output.color";
        resolveUses.push_back({currentHistory, Access::TextureStorageReadWrite});
        const std::array tonemapUses{RenderGraphStageUse{currentHistory, Access::TextureStorageReadWrite},
            RenderGraphStageUse{previousHistory, Access::TextureStorageRead},
            RenderGraphStageUse{"output.color", Access::TextureStorageWrite}};
        const std::array stages{
            RenderGraphStage{"NRC begin frame", sdkUses, [&](CommandBuffer& commands) -> Result<> {
                auto begun = nrc_.beginFrame(commands, frameSettings);
                return begun ? prepareParams() : begun;
            }, RenderGraphPassKind::Unsafe},
            RenderGraphStage{"NRC update", resources.uses, [&](CommandBuffer& commands) {
                return updateProgram.dispatch(commands, traceEncoded, (trainingWidth + 7) / 8, (trainingHeight + 7) / 8);
            }},
            RenderGraphStage{"NRC query", resources.uses, [&](CommandBuffer& commands) {
                return queryProgram.dispatch(commands, traceEncoded, (push.width + 7) / 8, (push.height + 7) / 8);
            }},
            RenderGraphStage{"NRC train", sdkUses,
                [&](CommandBuffer& commands) { return nrc_.queryAndTrain(commands, nullptr); }, RenderGraphPassKind::Unsafe},
            RenderGraphStage{"NRC resolve", resolveUses,
                [&](CommandBuffer& commands) { return nrc_.resolve(commands, *historyCurrentView); }, RenderGraphPassKind::Unsafe},
            RenderGraphStage{"NRC tonemap", tonemapUses, [&](CommandBuffer& commands) {
                return tonemapProgram_.dispatch(commands, *tonemapEncoded,
                    (push.width + 7u) / 8u, (push.height + 7u) / 8u);
            }}};
        const auto pending = nrcEndFramePending_;
        const auto discarded = nrcDiscarded_;
        result = commandBuffer.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
            [pending] { *pending = true; }, [discarded] { *discarded = true; }));
        if (!result) { return result; }
        result = context.executeStages(stages, resources.buffers, resources.textures);
        if (result) { *nrcDiscarded_ = false; }
        return result;
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

    bool useOpenPBRBsdf(const RenderGraphProperties& properties) const
    {
        const std::string bsdf = stringProperty(properties, "bsdf", "standard");
        return realtime_ || bsdf == "openpbr" || bsdf == "OpenPBR";
    }

    bool hasValuePrograms() const
    {
        const auto binding = sceneResources_.materialBinding();
        return binding && binding->values() && binding->values()->programCount() != 0;
    }

    bool materialBinningEnabled(const RenderGraphProperties& settings) const
    {
        // The existing five bins classify legacy factors. A Value program can
        // change them per hit, so use the general kernel until sparse M2 bins land.
        return !hasValuePrograms() && boolProperty(settings, "materialBinning", true);
    }

    bool validateMaterialTarget(std::string& log) const
    {
        if (hasValuePrograms() && (!useOpenPBRBsdf(properties()) || streamMaterials_ ||
                (visibilityDeferred_ && properties().value("lightingMode", "reference") == "realtime"))) {
            log = "Custom Value programs currently require OpenPBR PT or reference VBuffer shading (no StreamAsset/realtime specialization)";
            return false;
        }
        const auto generation = sceneResources_.materialGeneration();
        if (!generation) { log = "Missing material generation"; return false; }
        const auto target = visibilityDeferred_ ? MaterialEvaluationTarget::VisibilityBuffer :
            (!useOpenPBRBsdf(properties()) && METALLIC_HAS_RTXCR
                ? MaterialEvaluationTarget::RayHitWithFiber : MaterialEvaluationTarget::SurfaceRayHit);
        return generation->supports(target, log);
    }

    static uint32_t debugViewFromProperties(const RenderGraphProperties& properties)
    {
        const std::string view = stringProperty(properties, "debugView", "final");
        if (view == "geometryNormal") {
            return kScenePathTraceDebugViewGeometryNormal;
        }
        if (view == "shadingNormal") {
            return kScenePathTraceDebugViewShadingNormal;
        }
        if (view == "mappedNormal") {
            return kScenePathTraceDebugViewMappedNormal;
        }
        if (view == "tangent") {
            return kScenePathTraceDebugViewTangent;
        }
        if (view == "bitangent") {
            return kScenePathTraceDebugViewBitangent;
        }
        if (view == "tangentHandedness") {
            return kScenePathTraceDebugViewTangentHandedness;
        }
        if (view == "texcoord") {
            return kScenePathTraceDebugViewTexcoord;
        }
        if (view == "frontFace") {
            return kScenePathTraceDebugViewFrontFace;
        }
        if (view == "material") {
            return kScenePathTraceDebugViewMaterial;
        }
        if (view == "instance") {
            return kScenePathTraceDebugViewInstance;
        }
        if (view == "triangle") {
            return kScenePathTraceDebugViewTriangle;
        }
        if (view == "baseColor") {
            return kScenePathTraceDebugViewBaseColor;
        }
        if (view == "normalTexture") {
            return kScenePathTraceDebugViewNormalTexture;
        }
        if (view == "shadowTransmittance") {
            return kScenePathTraceDebugViewShadowTransmittance;
        }
        if (view == "shadingSide") {
            return kScenePathTraceDebugViewShadingSide;
        }
        return kScenePathTraceDebugViewFinal;
    }

    static std::string historyNameForContext(const RenderGraphExecutionContext& context, uint32_t cacheMode)
    {
        std::string name(kScenePathTraceHistoryPrefix);
        name += context.passName();
        name += ".accumulation";
        if (cacheMode == kScenePathTraceCacheModeNRC) {
            name += ".hdr";
        }
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
        outPush.debugView = debugViewFromProperties(properties);
        if (boolProperty(properties, "debugDisableNormalMap", false)) {
            outPush.debugFlags |= kScenePathTraceDebugDisableNormalMap;
        }
        if (boolProperty(properties, "debugForceGeometryNormal", false)) {
            outPush.debugFlags |= kScenePathTraceDebugForceGeometryNormal;
        }
        if (boolProperty(properties, "debugDisableMaterialTextures", false)) {
            outPush.debugFlags |= kScenePathTraceDebugDisableMaterialTextures;
        }
        if (boolProperty(properties, "debugDisableDirectLighting", false)) {
            outPush.debugFlags |= kScenePathTraceDebugDisableDirectLighting;
        }
        if (boolProperty(properties, "debugUseOpaqueShadows", false)) {
            outPush.debugFlags |= kScenePathTraceDebugUseOpaqueShadows;
        }
        if (boolProperty(properties, "debugDisableShadows", false)) {
            outPush.debugFlags |= kScenePathTraceDebugDisableShadows;
        }
        if (boolProperty(properties, "debugDisableVolumeAttenuation", false)) {
            outPush.debugFlags |= kScenePathTraceDebugDisableVolumeAttenuation;
        }
        if (boolProperty(properties, "debugDisableTransmission", false)) {
            outPush.debugFlags |= kScenePathTraceDebugDisableTransmission;
        }
        if (boolProperty(properties, "stochasticTextureFiltering", false)) {
            outPush.debugFlags |= kScenePathTraceDebugStochasticTextureFiltering;
        }

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

    ComputeKernel errorProgram_;
    std::vector<std::shared_ptr<const MaterialExecutableArtifact>> materialArtifacts_;
    bool realtime_ = false;
    bool visibilityDeferred_ = false;
    std::unique_ptr<PipelineCache> deferredPipelineCache_;
    MaterialBinning materialBinning_;
    ScreenSpaceShadows shadows_;
    std::array<float, 2> previousShadowJitter_{};
    std::array<ComputeKernel, kMaterialClassCount - 1> classifiedPrograms_;
    bool compiledMaterialBinning_ = true;
    bool compiledSupplementaryPathTracing_ = false;
    bool compiledRealtimeDeferred_ = false;
    bool compiledExportUpscalerGuides_ = false;
    RenderGraphProperties deferredHistoryProperties_;
    GPUSceneViewId deferredHistoryView_;
    uint64_t deferredRasterSettingsRevision_ = 0;
    SceneLightResources lights_;
    ScenePathTraceResources sceneResources_;
    SamplerDesc materialSampler_{
        .minFilter = SamplerFilter::Linear,
        .magFilter = SamplerFilter::Linear,
        .mipFilter = SamplerFilter::Linear,
        .addressU = SamplerAddressMode::Repeat,
        .addressV = SamplerAddressMode::Repeat,
        .addressW = SamplerAddressMode::Repeat,
    };
    bool streamMaterials_ = false;
    bool streamRayQueries_ = false;
    Device* device_ = nullptr;
    Queue* graphicsQueue_ = nullptr;
    OpenPBRLutResources openPBRLuts_;
    ComputeKernel typedPathTraceProgram_;
    std::array<ComputeKernel, 2> sharcTracePrograms_;
    std::array<ComputeKernel, 2> nrcTracePrograms_;
    ComputeKernel sharcClearProgram_;
    ComputeKernel sharcResolveProgram_;
    ComputeKernel tonemapProgram_;
    std::string compiledShaderKey_;
    uint32_t cacheMode_ = kScenePathTraceCacheModeOff;
    std::unique_ptr<Buffer> sharcHashEntriesBuffer_;
    std::unique_ptr<Buffer> sharcAccumulationBuffer_;
    std::unique_ptr<Buffer> sharcResolvedBuffer_;
    uint32_t sharcEntryCount_ = 0;
    uint64_t sharcResourcesRevision_ = 0;
    bool sharcClearPending_ = false;
    std::shared_ptr<bool> sharcDiscarded_ = std::make_shared<bool>(false);
#if METALLIC_HAS_NRC
    vulkan::NRCIntegration nrc_;
    nrc::ContextSettings nrcContextSettings_{};
    bool nrcConfigured_ = false;
    std::shared_ptr<bool> nrcEndFramePending_ = std::make_shared<bool>(false);
    std::shared_ptr<bool> nrcDiscarded_ = std::make_shared<bool>(false);
    uint64_t nrcSceneRevision_ = 0;
    uint64_t nrcEnvironmentRevision_ = 0;
#endif
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

std::unique_ptr<RenderGraphPass> createSceneRealtimeLightingPass()
{
    return std::make_unique<ScenePathTracePass>(true);
}

std::unique_ptr<RenderGraphPass> createVisibilityBufferDeferredPass()
{
    // Share material textures, OpenPBR LUTs, RT shadows and lighting with the
    // reference renderer; only the primary surface comes from raster visibility.
    return std::make_unique<ScenePathTracePass>(true, true);
}

} // namespace metallic::render::builtin_pass
