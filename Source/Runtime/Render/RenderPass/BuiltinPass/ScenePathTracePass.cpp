#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include "openpbr_data_constants.h"

#ifndef METALLIC_HAS_RTXCR
#define METALLIC_HAS_RTXCR 0
#endif

#ifndef METALLIC_RTXCR_SHADER_INCLUDE_DIR
#define METALLIC_RTXCR_SHADER_INCLUDE_DIR ""
#endif

namespace metallic::render::builtin_pass {
namespace {

using OpenPBRLutScalar = uint16_t;

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
constexpr uint32_t kEnvironmentImportanceAliasTableBinding = 13;
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
        const std::string shaderKey = std::string(moduleName) + "." + entryPointName;
        if (useOpenPBR) {
            result = openPBRLuts_.prepare(*context.device, log);
            if (!result) {
                rayQueryProgram_.clear();
                compiledShaderKey_.clear();
                return result;
            }
        }
        if (compiledShaderKey_ != shaderKey) {
            rayQueryProgram_.clear();
            compiledShaderKey_.clear();
            resetAccumulation_ = true;
            hasPreviousCamera_ = false;
        }
        if (rayQueryProgram_.valid()) {
            return {};
        }

        ShaderCompileResult computeCompile;
        const char* capabilities[] = {"spvRayQueryKHR"};
        const SlangMacroDefine macroDefines[] = {
            SlangMacroDefine{
                .name = "METALLIC_HAS_RTXCR",
                .value = METALLIC_HAS_RTXCR ? "1" : "0",
            },
        };
#if METALLIC_HAS_RTXCR
        const char* additionalSearchPaths[] = {METALLIC_RTXCR_SHADER_INCLUDE_DIR};
#endif
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = moduleName,
                .entryPointName = entryPointName,
                .searchPath = kTriangleShaderSearchPath,
#if METALLIC_HAS_RTXCR
                .additionalSearchPaths = additionalSearchPaths,
                .additionalSearchPathCount =
                    static_cast<uint32_t>(std::size(additionalSearchPaths)),
#endif
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
                .macroDefines = macroDefines,
                .macroDefineCount = static_cast<uint32_t>(std::size(macroDefines)),
            },
            computeCompile);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += moduleName;
            log += ".";
            log += entryPointName;
            log += ") returned ";
            log += resultToString(result);
            if (!computeCompile.diagnostics.empty()) {
                log += ": ";
                log += computeCompile.diagnostics;
            }
            log += '\n';
            rayQueryProgram_.clear();
            return result;
        }

        std::vector<SceneRayQueryBindingDesc> bindings{
            SceneRayQueryBindingDesc{
                .binding = 0,
                .kind = SceneRayQueryBindingKind::AccelerationStructure,
            },
            SceneRayQueryBindingDesc{
                .binding = 1,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 2,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 3,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 4,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 5,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 6,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 7,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 8,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 9,
                .kind = SceneRayQueryBindingKind::SampledImage,
                .descriptorCount = kScenePathTraceMaxMaterialTextures,
            },
            SceneRayQueryBindingDesc{
                .binding = 10,
                .kind = SceneRayQueryBindingKind::SampledImage,
            },
            SceneRayQueryBindingDesc{
                .binding = kEnvironmentImportanceAliasTableBinding,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
        };
        if (useOpenPBR) {
            bindings.push_back(SceneRayQueryBindingDesc{
                .binding = kOpenPBRLut2DBinding,
                .kind = SceneRayQueryBindingKind::SampledImage,
                .descriptorCount = kOpenPBRLut2DCount,
            });
            bindings.push_back(SceneRayQueryBindingDesc{
                .binding = kOpenPBRLut3DBinding,
                .kind = SceneRayQueryBindingKind::SampledImage,
                .descriptorCount = kOpenPBRLut3DCount,
            });
        }
        if (exportGuides) {
            bindings.push_back(SceneRayQueryBindingDesc{
                .binding = kDlssRrAlbedoBinding,
                .kind = SceneRayQueryBindingKind::StorageImage,
            });
            bindings.push_back(SceneRayQueryBindingDesc{
                .binding = kDlssRrSpecularAlbedoBinding,
                .kind = SceneRayQueryBindingKind::StorageImage,
            });
            bindings.push_back(SceneRayQueryBindingDesc{
                .binding = kDlssRrNormalRoughnessBinding,
                .kind = SceneRayQueryBindingKind::StorageImage,
            });
            bindings.push_back(SceneRayQueryBindingDesc{
                .binding = kDlssRrMotionVectorsBinding,
                .kind = SceneRayQueryBindingKind::StorageImage,
            });
            bindings.push_back(SceneRayQueryBindingDesc{
                .binding = kDlssRrLinearDepthBinding,
                .kind = SceneRayQueryBindingKind::StorageImage,
            });
            bindings.push_back(SceneRayQueryBindingDesc{
                .binding = kDlssRrSpecularHitDistanceBinding,
                .kind = SceneRayQueryBindingKind::StorageImage,
            });
        }
        std::string programLog;
        result = rayQueryProgram_.initialize(
            *context.device,
            SceneRayQueryProgramDesc{
                .spirv = computeCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(computeCompile.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(ScenePathTracePush),
                .bindings = bindings.data(),
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .debugName = useOpenPBR ? "ScenePathTracePass.OpenPBR" : "ScenePathTracePass",
            },
            programLog);
        if (!programLog.empty()) {
            if (!log.empty() && log.back() != '\n') {
                log += '\n';
            }
            log += programLog;
        }
        if (!result) {
            rayQueryProgram_.clear();
            return result;
        }
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
        Buffer* environmentImportanceBuffer = environment.importanceBuffer;
        TextureView* const environmentTextureViews[] = {environmentTextureView};
        const bool useOpenPBR = useOpenPBRBsdf(properties());
        const bool exportGuides = exportDenoiserGuides(properties());
        TextureHandle albedo = exportGuides ? context.outputTexture("albedo") : TextureHandle{};
        TextureHandle specularAlbedo = exportGuides ? context.outputTexture("specularAlbedo") : TextureHandle{};
        TextureHandle normalRoughness = exportGuides ? context.outputTexture("normalRoughness") : TextureHandle{};
        TextureHandle motionVectors = exportGuides ? context.outputTexture("motionVectors") : TextureHandle{};
        TextureHandle linearDepth = exportGuides ? context.outputTexture("linearDepth") : TextureHandle{};
        TextureHandle specularHitDistance = exportGuides ? context.outputTexture("specularHitDistance") : TextureHandle{};
        if (!color.valid() ||
            color.view() == nullptr ||
            !rayQueryProgram_.valid() ||
            !sceneResources_.valid() ||
            materialTextureViews[0] == nullptr ||
            environmentTextureView == nullptr ||
            environmentImportanceBuffer == nullptr ||
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
        push.environmentImportanceTexelCount = environment.importanceTexelCount;
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

        std::vector<SceneRayQueryDispatchBinding> bindings{
            SceneRayQueryDispatchBinding{
                .binding = 0,
                .accelerationStructure = &sceneResources_.accelerationStructure(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 1,
                .textureView = color.view(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 2,
                .buffer = sceneResources_.vertexBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 3,
                .buffer = sceneResources_.indexBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 4,
                .buffer = sceneResources_.primitiveBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 5,
                .buffer = sceneResources_.instanceBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 6,
                .buffer = sceneResources_.materialBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 7,
                .textureView = historyCurrentView,
            },
            SceneRayQueryDispatchBinding{
                .binding = 8,
                .textureView = historyPreviousView,
            },
            SceneRayQueryDispatchBinding{
                .binding = 9,
                .textureViews = materialTextureViews.data(),
                .textureViewCount = static_cast<uint32_t>(materialTextureViews.size()),
            },
            SceneRayQueryDispatchBinding{
                .binding = 10,
                .textureViews = environmentTextureViews,
                .textureViewCount = static_cast<uint32_t>(std::size(environmentTextureViews)),
            },
            SceneRayQueryDispatchBinding{
                .binding = kEnvironmentImportanceAliasTableBinding,
                .buffer = environmentImportanceBuffer,
            },
        };
        if (useOpenPBR) {
            const auto& lut2DViews = openPBRLuts_.lut2DViews();
            const auto& lut3DViews = openPBRLuts_.lut3DViews();
            bindings.push_back(SceneRayQueryDispatchBinding{
                .binding = kOpenPBRLut2DBinding,
                .textureViews = lut2DViews.data(),
                .textureViewCount = static_cast<uint32_t>(lut2DViews.size()),
            });
            bindings.push_back(SceneRayQueryDispatchBinding{
                .binding = kOpenPBRLut3DBinding,
                .textureViews = lut3DViews.data(),
                .textureViewCount = static_cast<uint32_t>(lut3DViews.size()),
            });
        }
        if (exportGuides) {
            bindings.push_back(SceneRayQueryDispatchBinding{
                .binding = kDlssRrAlbedoBinding,
                .textureView = albedo.view(),
            });
            bindings.push_back(SceneRayQueryDispatchBinding{
                .binding = kDlssRrSpecularAlbedoBinding,
                .textureView = specularAlbedo.view(),
            });
            bindings.push_back(SceneRayQueryDispatchBinding{
                .binding = kDlssRrNormalRoughnessBinding,
                .textureView = normalRoughness.view(),
            });
            bindings.push_back(SceneRayQueryDispatchBinding{
                .binding = kDlssRrMotionVectorsBinding,
                .textureView = motionVectors.view(),
            });
            bindings.push_back(SceneRayQueryDispatchBinding{
                .binding = kDlssRrLinearDepthBinding,
                .textureView = linearDepth.view(),
            });
            bindings.push_back(SceneRayQueryDispatchBinding{
                .binding = kDlssRrSpecularHitDistanceBinding,
                .textureView = specularHitDistance.view(),
            });
        }
        result = rayQueryProgram_.dispatch(SceneRayQueryDispatchDesc{
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
        const bool accumulationEnabled = boolProperty(context.properties(), "accumulate", true);
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

        const Format historyFormat = exportDenoiserGuides(context.properties())
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
        const std::string historyName = historyNameForContext(context);
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
    SceneRayQueryProgram rayQueryProgram_;
    std::string compiledShaderKey_;
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
