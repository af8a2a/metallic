#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderGraph/NrdRuntime.h"
#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/ReGIR.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#ifndef METALLIC_HAS_NTC
#define METALLIC_HAS_NTC 0
#endif

#ifndef METALLIC_NTC_SHADER_INCLUDE_DIR
#define METALLIC_NTC_SHADER_INCLUDE_DIR ""
#endif

namespace metallic::render::builtin_pass {
namespace {

struct SceneRtxdiCameraSnapshot {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
};

struct SceneRtxdiHistoryViews {
    TextureView* current = nullptr;
    TextureView* previous = nullptr;
    bool previousValid = false;
};

class SceneRtxdiPass final : public ComputePass {
public:
    ~SceneRtxdiPass() override = default;

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{
            EnvironmentLightingSubsystem::kSubsystemId,
        };
        return required;
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "ReSTIR DI many-light direct illumination")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        reflection.addTextureOutput("noisyDiffuse", "RELAX diffuse radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("noisySpecular", "RELAX specular radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("normalRoughness", "NRD packed world normal and roughness")
            .storageReadWrite()
            .format = nrdNormalRoughnessFormat();
        reflection.addTextureOutput("motionVectors", "NRD previous-minus-current UV motion")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("viewZ", "NRD linear view depth")
            .storageReadWrite()
            .format = Format::R16Sfloat;
        reflection.addTextureOutput("baseColorMetalness", "NRD base color and metalness")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        reflection.addTextureOutput("emissive", "Emissive and background radiance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeIntSetting(
                "lightCount",
                "Analytic Lights",
                static_cast<int32_t>(kDefaultRtxdiLightCount),
                1,
                static_cast<int32_t>(kMaxRtxdiLightCount),
                true),
            runtimeIntSetting(
                "initialSamples",
                "Initial Candidates",
                static_cast<int32_t>(kDefaultRtxdiInitialSamples),
                1,
                static_cast<int32_t>(kMaxRtxdiInitialSamples),
                true),
            runtimeBoolSetting(
                "localLightImportanceSampling",
                "Local Light PDF",
                true,
                true),
            runtimeBoolSetting("regirEnabled", "ReGIR Selector", true, true),
            runtimeIntSetting(
                "regirGridSize",
                "ReGIR Grid Resolution",
                static_cast<int32_t>(kDefaultReGIRGridSize),
                4,
                static_cast<int32_t>(kMaxReGIRGridSize),
                true),
            runtimeIntSetting(
                "regirLightsPerCell",
                "ReGIR Lights per Cell",
                static_cast<int32_t>(kDefaultReGIRLightsPerCell),
                8,
                static_cast<int32_t>(kMaxReGIRLightsPerCell),
                true),
            runtimeIntSetting(
                "regirBuildSamples",
                "ReGIR Build Samples",
                static_cast<int32_t>(kDefaultReGIRBuildSamples),
                1,
                static_cast<int32_t>(kMaxReGIRBuildSamples),
                true),
            runtimeFloatSetting(
                "regirSamplingJitter",
                "ReGIR Sampling Jitter",
                1.0f,
                0.0f,
                2.0f,
                true),
            runtimeIntSetting(
                "environmentSamples",
                "Environment Candidates",
                4,
                0,
                16,
                true),
            runtimeBoolSetting(
                "environmentImportanceSampling",
                "Environment PDF",
                true,
                true),
            runtimeIntSetting(
                "spatialSamples",
                "Spatial Neighbors",
                static_cast<int32_t>(kDefaultRtxdiSpatialSamples),
                0,
                static_cast<int32_t>(kMaxRtxdiSpatialSamples),
                true),
            runtimeIntSetting("maxHistoryLength", "History Length", 20, 1, 64, true),
            runtimeBoolSetting("temporalReuse", "Temporal Reuse", true, true),
            runtimeBoolSetting("spatialReuse", "Spatial Reuse", true, true),
            runtimeBoolSetting("initialVisibility", "Initial Visibility", true, true),
            runtimeBoolSetting("animateLights", "Animate Lights", true, true),
            runtimeFloatSetting("lightIntensity", "Light Intensity", 12.0f, 0.0f, 100.0f, true),
            runtimeFloatSetting("exposure", "Exposure", 1.0f, 0.05f, 8.0f),
            runtimeEnumSetting(
                "visualization",
                "Visualization",
                "shaded",
                {
                    {"Shaded", "shaded"},
                    {"Selected Light", "lightId"},
                    {"Reservoir History", "history"},
                    {"ReGIR Cells", "regirCells"},
                }),
            runtimeBoolSetting("flipBitangent", "Flip Bitangent", false, true),
        };
        appendCameraRuntimeSettings(
            settings,
            std::array<float, 3>{0.0f, 0.25f, 3.0f},
            std::array<float, 3>{0.0f, 0.15f, 0.0f},
            50.0f,
            true);
        return settings;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "SceneRtxdiPass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().rayTracingAccelerationStructure ||
            !context.device->capabilities().rayQuery) {
            log = "SceneRtxdiPass requires rayTracingAccelerationStructure and rayQuery capabilities";
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
            resetHistory_ = true;
            hasPreviousCamera_ = false;
        }
        result = importancePdfCompute_.initialize(*context.device, log);
        if (!result) {
            return result;
        }
        result = reGIR_.initialize(*context.device, log);
        if (!result) {
            return result;
        }
        result = ensureImportancePdfResources(
            *context.device,
            uintProperty(properties(), "lightCount", kDefaultRtxdiLightCount, 1, kMaxRtxdiLightCount),
            log);
        if (!result) {
            return result;
        }
        result = ensureReGIRResources(*context.device, properties(), log);
        if (!result) {
            return result;
        }
        const bool ntcActive = sceneResources_.neuralTextures().active();
        if (rayQueryProgram_.valid() && compiledNtcActive_ == ntcActive) {
            return {};
        }
        rayQueryProgram_.clear();

        ShaderCompileResult computeCompile;
        const char* capabilities[] = {"spvRayQueryKHR"};
        const SlangMacroDefine defines[] = {
            SlangMacroDefine{
                .name = "METALLIC_HAS_NTC",
                .value = ntcActive ? "1" : "0",
            },
        };
        std::vector<const char*> additionalSearchPaths;
#if METALLIC_HAS_NTC
        if (ntcActive) {
            additionalSearchPaths.push_back(METALLIC_NTC_SHADER_INCLUDE_DIR);
        }
#endif
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kSceneRtxdiShaderModuleName,
                .entryPointName = kSceneRtxdiEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .additionalSearchPaths = additionalSearchPaths.data(),
                .additionalSearchPathCount =
                    static_cast<uint32_t>(additionalSearchPaths.size()),
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
                .macroDefines = defines,
                .macroDefineCount = static_cast<uint32_t>(std::size(defines)),
            },
            computeCompile);
        if (!result) {
            log += "compileSlangShaderToSpirv(SceneRtxdi.sceneRtxdiMain) returned ";
            log += resultToString(result);
            if (!computeCompile.diagnostics.empty()) {
                log += ": ";
                log += computeCompile.diagnostics;
            }
            log += '\n';
            rayQueryProgram_.clear();
            return result;
        }

        // Keep the conventional binding table stable; append NTC descriptors only when active.
        std::vector<ComputeProgramBindingDesc> bindings{
            {.binding = 0, .kind = ComputeResourceBindingKind::AccelerationStructure},
            {.binding = 1, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 2, .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = 3, .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = 4, .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = 5, .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = 6, .kind = ComputeResourceBindingKind::StorageBuffer},
            {
                .binding = 7,
                .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = kScenePathTraceMaxMaterialTextures,
            },
            {.binding = 8, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 9, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 10, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 11, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 12, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 13, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 14, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 15, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 16, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 17, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 18, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 19, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 20, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 21, .kind = ComputeResourceBindingKind::SampledImage},
            {.binding = 22, .kind = ComputeResourceBindingKind::SampledImage},
            {.binding = 23, .kind = ComputeResourceBindingKind::SampledImage},
            {.binding = 24, .kind = ComputeResourceBindingKind::StorageBuffer},
        };
        if (ntcActive) {
            bindings.push_back({
                .binding = kNeuralTextureLatentsBinding,
                .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = kMaxNeuralTextureSets,
            });
            bindings.push_back({
                .binding = kNeuralTextureConstantsBinding,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            bindings.push_back({
                .binding = kNeuralTextureWeightsBinding,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            bindings.push_back({
                .binding = kNeuralTextureSetInfoBinding,
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            bindings.push_back({
                .binding = kNeuralTextureSamplerBinding,
                .kind = ComputeResourceBindingKind::Sampler,
            });
        }
        std::string programLog;
        result = rayQueryProgram_.initialize(
            *context.device,
            ComputeProgramDesc{
                .spirv = computeCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(computeCompile.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(SceneRtxdiPush),
                .bindings = bindings.data(),
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .debugName = "SceneRtxdiPass",
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
        } else {
            compiledNtcActive_ = ntcActive;
        }
        return result;
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
            spdlog::warn("[SceneRtxdiPass] Runtime scene sync failed: {}", syncLog);
            return syncResult;
        }
        if (!sceneResources_.textureUploadsReady()) {
            return {};
        }
        if (sceneResources_.revision() != sceneResourceRevision_) {
            sceneResourceRevision_ = sceneResources_.revision();
            resetHistory_ = true;
            hasPreviousCamera_ = false;
        }
        if (device_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        std::string importanceLog;
        Result importanceResult = ensureImportancePdfResources(
            *device_,
            uintProperty(
                context.properties(),
                "lightCount",
                kDefaultRtxdiLightCount,
                1,
                kMaxRtxdiLightCount),
            importanceLog);
        if (!importanceResult) {
            spdlog::warn("[SceneRtxdiPass] Local light PDF rebuild failed: {}", importanceLog);
            return importanceResult;
        }
        Result reGIRResult = ensureReGIRResources(*device_, context.properties(), importanceLog);
        if (!reGIRResult) {
            spdlog::warn("[SceneRtxdiPass] ReGIR rebuild failed: {}", importanceLog);
            return reGIRResult;
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
            resetHistory_ = true;
            hasPreviousCamera_ = false;
        }
        TextureHandle color = context.outputTexture("color");
        TextureHandle noisyDiffuse = context.outputTexture("noisyDiffuse");
        TextureHandle noisySpecular = context.outputTexture("noisySpecular");
        TextureHandle normalRoughness = context.outputTexture("normalRoughness");
        TextureHandle motionVectors = context.outputTexture("motionVectors");
        TextureHandle viewZ = context.outputTexture("viewZ");
        TextureHandle baseColorMetalness = context.outputTexture("baseColorMetalness");
        TextureHandle emissive = context.outputTexture("emissive");
        const auto& materialTextureViews = sceneResources_.materialTextureViews();
        TextureView* environmentTextureView = environment.radianceView;
        TextureView* environmentImportanceTextureView = environment.pdfView;
        if (!validTexture(color) ||
            !validTexture(noisyDiffuse) ||
            !validTexture(noisySpecular) ||
            !validTexture(normalRoughness) ||
            !validTexture(motionVectors) ||
            !validTexture(viewZ) ||
            !validTexture(baseColorMetalness) ||
            !validTexture(emissive) ||
            !rayQueryProgram_.valid() ||
            !sceneResources_.valid() ||
            materialTextureViews[0] == nullptr ||
            environmentTextureView == nullptr ||
            environmentImportanceTextureView == nullptr ||
            !localLightPdf_.valid() ||
            localLightPdf_.view() == nullptr ||
            !importancePdfCompute_.valid() ||
            !reGIR_.valid() ||
            reGIR_.buffer() == nullptr ||
            context.historyResources() == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        SceneRtxdiHistoryViews reservoirHistory;
        SceneRtxdiHistoryViews positionHistory;
        SceneRtxdiHistoryViews normalHistory;
        Result result = prepareHistoryTexture(
            context,
            "reservoir",
            Format::Rgba32Uint,
            reservoirHistory);
        if (!result) {
            return result;
        }
        result = prepareHistoryTexture(
            context,
            "position",
            Format::Rgba32Sfloat,
            positionHistory);
        if (!result) {
            return result;
        }
        result = prepareHistoryTexture(
            context,
            "normal",
            Format::Rgba16Sfloat,
            normalHistory);
        if (!result) {
            return result;
        }

        SceneRtxdiPush push;
        buildPush(
            context.width(),
            context.height(),
            context.properties(),
            sceneResources_.bounds(),
            environment.settings,
            push);
        push.materialTextureCount = sceneResources_.materialTextureCount();
        push.ntcTextureSetCount = sceneResources_.neuralTextures().textureSetCount();
        if (!environment.mapAvailable) {
            push.behaviorFlags &= ~kRtxdiBehaviorEnvironmentEnabled;
        }
        push.frameIndex = frameIndex_++;

        const SceneRtxdiCameraSnapshot currentCamera = cameraSnapshotFromPush(push);
        const bool cameraHistoryValid =
            hasPreviousCamera_ &&
            previousCameraWidth_ == context.width() &&
            previousCameraHeight_ == context.height();
        const bool historyValid =
            !resetHistory_ &&
            cameraHistoryValid &&
            reservoirHistory.previousValid &&
            positionHistory.previousValid &&
            normalHistory.previousValid;
        applyPreviousCameraSnapshot(historyValid ? previousCamera_ : currentCamera, push);
        push.hasHistory = historyValid ? 1u : 0u;

        result = sceneResources_.uploadMaterialTextures(context.commandBuffer());
        if (!result) {
            return result;
        }
        result = importancePdfCompute_.buildLocalLights(
            context.commandBuffer(),
            *environmentTextureView,
            localLightPdf_,
            push.lightCount,
            push.lightIntensity,
            push.sceneCenterRadius[3]);
        if (!result) {
            return result;
        }

        if ((push.behaviorFlags & kRtxdiBehaviorReGIR) != 0u) {
            ReGIRBuildParameters reGIRBuild;
            reGIRBuild.lightCount = push.lightCount;
            reGIRBuild.buildSamples = uintProperty(
                context.properties(),
                "regirBuildSamples",
                kDefaultReGIRBuildSamples,
                1,
                kMaxReGIRBuildSamples);
            reGIRBuild.frameIndex = push.frameIndex;
            reGIRBuild.animateLights =
                (push.behaviorFlags & kRtxdiBehaviorAnimateLights) != 0u;
            reGIRBuild.sceneCenter[0] = push.sceneCenterRadius[0];
            reGIRBuild.sceneCenter[1] = push.sceneCenterRadius[1];
            reGIRBuild.sceneCenter[2] = push.sceneCenterRadius[2];
            reGIRBuild.sceneRadius = push.sceneCenterRadius[3];
            reGIRBuild.lightIntensity = push.lightIntensity;
            reGIRBuild.samplingJitter = std::max(
                floatProperty(context.properties(), "regirSamplingJitter", 1.0f),
                0.0f);
            result = reGIR_.build(
                context.commandBuffer(),
                *localLightPdf_.view(),
                reGIRBuild);
            if (!result) {
                return result;
            }
        }

        TextureView* const environmentTextureViews[] = {environmentTextureView};
        TextureView* const localLightPdfViews[] = {localLightPdf_.view()};
        TextureView* const environmentImportanceTextureViews[] = {environmentImportanceTextureView};
        std::vector<ComputeDispatchBinding> bindings{
            {
                .binding = 0,
                .accelerationStructure =
                    sceneResources_.accelerationStructure().accelerationStructure(),
            },
            {.binding = 1, .textureView = color.view()},
            {.binding = 2, .buffer = sceneResources_.vertexBuffer()},
            {.binding = 3, .buffer = sceneResources_.indexBuffer()},
            {.binding = 4, .buffer = sceneResources_.primitiveBuffer()},
            {.binding = 5, .buffer = sceneResources_.instanceBuffer()},
            {.binding = 6, .buffer = sceneResources_.materialBuffer()},
            {
                .binding = 7,
                .textureViews = materialTextureViews.data(),
                .textureViewCount = static_cast<uint32_t>(materialTextureViews.size()),
            },
            {.binding = 8, .textureView = reservoirHistory.current},
            {.binding = 9, .textureView = reservoirHistory.previous},
            {.binding = 10, .textureView = positionHistory.current},
            {.binding = 11, .textureView = positionHistory.previous},
            {.binding = 12, .textureView = normalHistory.current},
            {.binding = 13, .textureView = normalHistory.previous},
            {.binding = 14, .textureView = noisyDiffuse.view()},
            {.binding = 15, .textureView = noisySpecular.view()},
            {.binding = 16, .textureView = normalRoughness.view()},
            {.binding = 17, .textureView = motionVectors.view()},
            {.binding = 18, .textureView = viewZ.view()},
            {.binding = 19, .textureView = baseColorMetalness.view()},
            {.binding = 20, .textureView = emissive.view()},
            {
                .binding = 21,
                .textureViews = environmentTextureViews,
                .textureViewCount = static_cast<uint32_t>(std::size(environmentTextureViews)),
            },
            {
                .binding = 22,
                .textureViews = localLightPdfViews,
                .textureViewCount = static_cast<uint32_t>(std::size(localLightPdfViews)),
            },
            {
                .binding = 23,
                .textureViews = environmentImportanceTextureViews,
                .textureViewCount = static_cast<uint32_t>(std::size(environmentImportanceTextureViews)),
            },
            {.binding = 24, .buffer = reGIR_.buffer()},
        };
        const NeuralTextureResources& neuralTextures = sceneResources_.neuralTextures();
        if (neuralTextures.active()) {
            const auto& latentViews = neuralTextures.latentTextureViews();
            bindings.push_back({
                .binding = kNeuralTextureLatentsBinding,
                .textureViews = latentViews.data(),
                .textureViewCount = static_cast<uint32_t>(latentViews.size()),
            });
            bindings.push_back({
                .binding = kNeuralTextureConstantsBinding,
                .buffer = neuralTextures.constantsBuffer(),
            });
            bindings.push_back({
                .binding = kNeuralTextureWeightsBinding,
                .buffer = neuralTextures.weightsBuffer(),
            });
            bindings.push_back({
                .binding = kNeuralTextureSetInfoBinding,
                .buffer = neuralTextures.setInfoBuffer(),
            });
            bindings.push_back({
                .binding = kNeuralTextureSamplerBinding,
                .sampler = &neuralTextures.latentSampler(),
            });
        }
        result = rayQueryProgram_.dispatch(ComputeDispatchDesc{
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

        HistoryResourceManager& history = *context.historyResources();
        history.markWritten(historyNameForContext(context, "reservoir"));
        history.markWritten(historyNameForContext(context, "position"));
        history.markWritten(historyNameForContext(context, "normal"));
        previousCamera_ = currentCamera;
        previousCameraWidth_ = context.width();
        previousCameraHeight_ = context.height();
        hasPreviousCamera_ = true;
        resetHistory_ = false;
        return {};
    }

private:
    Result ensureImportancePdfResources(
        Device& device,
        uint32_t lightCount,
        std::string& log)
    {
        if (!localLightPdf_.valid() || localLightPdfLightCount_ != lightCount) {
            const ImportancePdfSize localPdfSize = computeImportancePdfTextureSize(lightCount);
            ImportancePdfTexture nextPdf;
            Result result = nextPdf.initialize(
                device,
                localPdfSize.width,
                localPdfSize.height,
                "SceneRtxdiPass local light importance PDF",
                log);
            if (!result) {
                return result;
            }
            if (localLightPdf_.valid()) {
                retiredLocalLightPdfs_.push_back(std::move(localLightPdf_));
            }
            localLightPdf_ = std::move(nextPdf);
            localLightPdfLightCount_ = lightCount;
            resetHistory_ = true;
        }

        return {};
    }

    Result ensureReGIRResources(
        Device& device,
        const RenderGraphProperties& properties,
        std::string& log)
    {
        const uint32_t gridSize = uintProperty(
            properties,
            "regirGridSize",
            kDefaultReGIRGridSize,
            4,
            kMaxReGIRGridSize);
        const uint32_t lightsPerCell = uintProperty(
            properties,
            "regirLightsPerCell",
            kDefaultReGIRLightsPerCell,
            8,
            kMaxReGIRLightsPerCell);
        const bool layoutChanged =
            reGIR_.layout().gridSize != gridSize ||
            reGIR_.layout().lightsPerCell != lightsPerCell;
        Result result = reGIR_.ensureGrid(device, gridSize, lightsPerCell, log);
        if (result && layoutChanged) {
            resetHistory_ = true;
        }
        return result;
    }

    static bool validTexture(TextureHandle texture)
    {
        return texture.valid() && texture.texture() != nullptr && texture.view() != nullptr;
    }

    static Result prepareHistoryTexture(
        RenderGraphExecutionContext& context,
        std::string_view suffix,
        Format format,
        SceneRtxdiHistoryViews& outViews)
    {
        HistoryResourceManager* history = context.historyResources();
        if (history == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        const TextureDesc desc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Storage | TextureUsageBits::TransferSource,
            .format = format,
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        };
        const std::string name = historyNameForContext(context, suffix);
        Result result = history->ensureTexture(name, desc, TextureViewDesc{.format = format});
        if (!result) {
            return result;
        }

        const HistoryTextureRef current = history->texture(name, HistorySlot::Current);
        const HistoryTextureRef previous = history->texture(name, HistorySlot::Previous);
        if (current.texture == nullptr || current.view == nullptr ||
            previous.texture == nullptr || previous.view == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        result = history->transitionTexture(
            context.commandBuffer(),
            name,
            HistorySlot::Current,
            ResourceState::General);
        if (!result) {
            return result;
        }
        result = history->transitionTexture(
            context.commandBuffer(),
            name,
            HistorySlot::Previous,
            ResourceState::General);
        if (!result) {
            return result;
        }
        outViews.current = current.view;
        outViews.previous = previous.view;
        outViews.previousValid = previous.valid;
        return {};
    }

    static std::string historyNameForContext(
        const RenderGraphExecutionContext& context,
        std::string_view suffix)
    {
        std::string name("SceneRtxdiPass.");
        name += context.passName();
        name += '.';
        name += suffix;
        return name;
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
        int64_t value = static_cast<int64_t>(fallback);
        if (iter->is_number_integer()) {
            value = iter->get<int64_t>();
        } else {
            value = static_cast<int64_t>(iter->get<double>());
        }
        return static_cast<uint32_t>(std::clamp<int64_t>(value, minimum, maximum));
    }

    static float finiteOr(float value, float fallback)
    {
        return std::isfinite(value) ? value : fallback;
    }

    static float floatProperty(
        const RenderGraphProperties& properties,
        const char* key,
        float fallback)
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

    static const RenderGraphProperties* cameraPropertiesFrom(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return nullptr;
        }
        auto iter = properties.find("camera");
        return iter != properties.end() && iter->is_object() ? &(*iter) : nullptr;
    }

    static float cameraFloat(const RenderGraphProperties* camera, const char* key, float fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        return iter != camera->end() && iter->is_number()
            ? finiteOr(iter->get<float>(), fallback)
            : fallback;
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
            if ((*iter)[index].is_number()) {
                values[index] = finiteOr((*iter)[index].get<float>(), values[index]);
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

    static uint32_t visualizationModeFromProperties(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return kRtxdiVisualizationShaded;
        }
        auto iter = properties.find("visualization");
        if (iter == properties.end() || !iter->is_string()) {
            return kRtxdiVisualizationShaded;
        }
        const std::string mode = iter->get<std::string>();
        if (mode == "lightId" || mode == "light" || mode == "selectedLight") {
            return kRtxdiVisualizationLightId;
        }
        if (mode == "history" || mode == "reservoirHistory") {
            return kRtxdiVisualizationHistory;
        }
        if (mode == "regirCells" || mode == "regir") {
            return kRtxdiVisualizationReGIRCells;
        }
        return kRtxdiVisualizationShaded;
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

    static SceneRtxdiCameraSnapshot cameraSnapshotFromPush(const SceneRtxdiPush& push)
    {
        SceneRtxdiCameraSnapshot snapshot;
        copyFloat4(push.eye, snapshot.eye);
        copyFloat4(push.center, snapshot.center);
        copyFloat4(push.upProjection, snapshot.upProjection);
        copyFloat4(push.viewport, snapshot.viewport);
        copyFloat4(push.clipOrtho, snapshot.clipOrtho);
        return snapshot;
    }

    static void applyPreviousCameraSnapshot(
        const SceneRtxdiCameraSnapshot& snapshot,
        SceneRtxdiPush& push)
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
        SceneRtxdiPush& outPush)
    {
        outPush = SceneRtxdiPush{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* camera = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovDegrees = std::clamp(cameraFloat(camera, "fovDegrees", 50.0f), 1.0f, 179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + radius
                : radius * 2.5f,
            0.05f);
        const float3 defaultEye(center.x, center.y + radius * 0.45f, center.z + defaultDistance);
        const float3 eye = cameraVec3(camera, "eye", defaultEye);
        const float3 target = cameraVec3(camera, "center", center);
        const float3 up = cameraVec3(camera, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(camera, "znear", 0.001f), 0.0001f);
        const float zFar = std::max(
            cameraFloat(camera, "zfar", defaultDistance + radius * 4.0f),
            zNear + 0.001f);
        const float cameraDistance = std::max(length(eye - target), 0.001f);
        const float defaultOrthoHeight = std::max(2.0f * cameraDistance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float orthoHeight = std::max(cameraFloat(camera, "orthoHeight", defaultOrthoHeight), 0.0001f);

        writeParamVec3(eye, outPush.eye, 0.0f);
        writeParamVec3(target, outPush.center, 0.0f);
        writeParamVec3(up, outPush.upProjection, cameraIsOrthographic(camera) ? 1.0f : 0.0f);
        outPush.viewport[0] = aspect;
        outPush.viewport[1] = static_cast<float>(width);
        outPush.viewport[2] = static_cast<float>(height);
        outPush.viewport[3] = fovRadians;
        outPush.clipOrtho[0] = zNear;
        outPush.clipOrtho[1] = zFar;
        outPush.clipOrtho[2] = orthoHeight;
        writeParamVec3(center, outPush.sceneCenterRadius, radius);
        outPush.width = width;
        outPush.height = height;
        outPush.lightCount = uintProperty(properties, "lightCount", kDefaultRtxdiLightCount, 1, kMaxRtxdiLightCount);
        outPush.initialSampleCount = uintProperty(
            properties,
            "initialSamples",
            kDefaultRtxdiInitialSamples,
            1,
            kMaxRtxdiInitialSamples);
        outPush.spatialSampleCount = uintProperty(
            properties,
            "spatialSamples",
            kDefaultRtxdiSpatialSamples,
            0,
            kMaxRtxdiSpatialSamples);
        outPush.maxHistoryLength = uintProperty(properties, "maxHistoryLength", 20, 1, 64);
        outPush.behaviorFlags = 0u;
        auto setBehavior = [&outPush](uint32_t flag, bool enabled) {
            if (enabled) {
                outPush.behaviorFlags |= flag;
            }
        };
        setBehavior(kRtxdiBehaviorTemporalReuse, boolProperty(&properties, "temporalReuse", true));
        setBehavior(kRtxdiBehaviorSpatialReuse, boolProperty(&properties, "spatialReuse", true));
        setBehavior(kRtxdiBehaviorInitialVisibility, boolProperty(&properties, "initialVisibility", true));
        setBehavior(kRtxdiBehaviorAnimateLights, boolProperty(&properties, "animateLights", true));
        outPush.behaviorFlags |=
            visualizationModeFromProperties(properties) << kRtxdiBehaviorVisualizationShift;
        outPush.bitangentFlip = boolProperty(&properties, "flipBitangent", false) ? -1.0f : 1.0f;
        outPush.lightIntensity = std::max(floatProperty(properties, "lightIntensity", 12.0f), 0.0f);
        outPush.exposure = std::max(floatProperty(properties, "exposure", 1.0f), 0.001f);
        setBehavior(
            kRtxdiBehaviorLocalLightImportance,
            boolProperty(&properties, "localLightImportanceSampling", true));
        setBehavior(kRtxdiBehaviorReGIR, boolProperty(&properties, "regirEnabled", true));
        outPush.environmentSampleCount = uintProperty(properties, "environmentSamples", 4, 0, 16);
        setBehavior(
            kRtxdiBehaviorEnvironmentImportance,
            boolProperty(&properties, "environmentImportanceSampling", true));
        setBehavior(
            kRtxdiBehaviorEnvironmentEnabled,
            environment.enabled);
        setBehavior(
            kRtxdiBehaviorEnvironmentVisible,
            environment.visible);
        outPush.environmentIntensity = std::max(environment.intensity, 0.0f);
        outPush.environmentRotationRadians = environment.rotationDegrees * (kPi / 180.0f);
    }

    ScenePathTraceResources sceneResources_;
    ComputeProgram rayQueryProgram_;
    bool compiledNtcActive_ = false;
    Device* device_ = nullptr;
    Queue* graphicsQueue_ = nullptr;
    SceneResourceManager* sceneResourceManager_ = nullptr;
    ImportancePdfCompute importancePdfCompute_;
    ImportancePdfTexture localLightPdf_;
    ReGIRLightSelector reGIR_;
    std::vector<ImportancePdfTexture> retiredLocalLightPdfs_;
    uint32_t localLightPdfLightCount_ = 0;
    uint64_t sceneResourceRevision_ = 0;
    uint64_t environmentResourceRevision_ = 0;
    uint64_t environmentSettingsRevision_ = 0;
    uint32_t frameIndex_ = 0;
    SceneRtxdiCameraSnapshot previousCamera_;
    uint32_t previousCameraWidth_ = 0;
    uint32_t previousCameraHeight_ = 0;
    bool hasPreviousCamera_ = false;
    bool resetHistory_ = true;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSceneRtxdiPass()
{
    return std::make_unique<SceneRtxdiPass>();
}

} // namespace metallic::render::builtin_pass
