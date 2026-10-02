#include "Runtime/Render/Core/StreamInstanceCullParameters.h"
#include "Runtime/Render/Core/HZBParameters.h"
#include "Runtime/Render/Core/StreamDeferredParameters.h"
#include "Runtime/Render/Core/StreamCompositeParameters.h"
#include "Runtime/Render/Core/DebugVisualizationParameters.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Subsystem/GPUSceneLightFrustum.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace metallic::render::builtin_pass {
namespace {

std::filesystem::path pathFromProperties(
    const RenderGraphProperties& props,
    const char* key,
    const std::filesystem::path& fallback)
{
    if (props.contains(key) && props[key].is_string()) {
        std::filesystem::path path = props[key].get<std::string>();
        if (path.is_relative()) {
            path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
        }
        return path;
    }
    return fallback;
}

std::filesystem::path scenePathFromProperties(const RenderGraphProperties& props)
{
    return pathFromProperties(props, "path", kDefaultGPUDrivenScenePath);
}

bool boolProperty(const RenderGraphProperties& props, const char* key, bool fallback)
{
    auto iter = props.find(key);
    return iter != props.end() && iter->is_boolean() ? iter->get<bool>() : fallback;
}

uint32_t uintProperty(const RenderGraphProperties& props, const char* key, uint32_t fallback)
{
    auto iter = props.find(key);
    if (iter == props.end() || !iter->is_number_integer()) {
        return fallback;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 || value > std::numeric_limits<uint32_t>::max()
        ? fallback
        : static_cast<uint32_t>(value);
}

uint32_t selectedLodProperty(const RenderGraphProperties& props)
{
    auto iter = props.find("lodLevel");
    if (iter == props.end() || !iter->is_number_integer()) {
        iter = props.find("selectedLodLevel");
    }
    if (iter == props.end() || !iter->is_number_integer()) {
        return 0;
    }
    const int64_t value = iter->get<int64_t>();
    return static_cast<uint32_t>(std::clamp<int64_t>(value, 0, 31));
}

bool autoLodProperty(const RenderGraphProperties& props)
{
    return boolProperty(props, "autoLod", boolProperty(props, "enableGpuLodSelection", true));
}

uint32_t debugColorModeFromProperties(const RenderGraphProperties& props)
{
    auto iter = props.find("debugColorMode");
    if (iter == props.end() || !iter->is_string()) {
        return kMeshletStreamDebugShaded;
    }
    const std::string mode = iter->get<std::string>();
    if (mode == "lod") {
        return kMeshletStreamDebugLOD;
    }
    if (mode == "page") {
        return kMeshletStreamDebugPage;
    }
    if (mode == "primitive") {
        return kMeshletStreamDebugPrimitive;
    }
    if (mode == "instance") {
        return kMeshletStreamDebugInstance;
    }
    if (mode == "meshlet" || mode == "cluster") {
        return kMeshletStreamDebugMeshlet;
    }
    if (mode == "shaded") {
        return kMeshletStreamDebugShaded;
    }
    return kMeshletStreamDebugShaded;
}

uint32_t rtasGranularityFromProperties(const RenderGraphProperties& props)
{
    auto iter = props.find("rtasGranularity");
    if (iter == props.end() || !iter->is_string()) {
        return kRayQueryVisualizationGranularityInstance;
    }
    const std::string mode = iter->get<std::string>();
    if (mode == "primitive") {
        return kRayQueryVisualizationGranularityPrimitive;
    }
    if (mode == "cluster" || mode == "cluster-id" || mode == "meshlet") {
        return kRayQueryVisualizationGranularityClusterId;
    }
    return kRayQueryVisualizationGranularityInstance;
}

uint32_t cullingFlagsFromProperties(const RenderGraphProperties& props)
{
    uint32_t flags = 0;
    if (boolProperty(props, "instanceFrustumCull", true)) {
        flags |= 1u << 0u;
    }
    if (boolProperty(props, "instanceHzbCull", true)) {
        flags |= 1u << 1u;
    }
    if (boolProperty(props, "clusterFrustumCull", true)) {
        flags |= 1u << 2u;
    }
    if (boolProperty(props, "clusterNormalConeCull", true)) {
        flags |= 1u << 3u;
    }
    return flags;
}

const RenderGraphProperties* cameraPropertiesFrom(const RenderGraphProperties& properties)
{
    auto iter = properties.find("camera");
    return iter != properties.end() && iter->is_object() ? &(*iter) : nullptr;
}

float finiteOr(float value, float fallback)
{
    return std::isfinite(value) ? value : fallback;
}

float cameraFloat(const RenderGraphProperties* camera, const char* key, float fallback)
{
    if (camera == nullptr) {
        return fallback;
    }
    auto iter = camera->find(key);
    return iter != camera->end() && iter->is_number()
        ? finiteOr(iter->get<float>(), fallback)
        : fallback;
}

float3 cameraVec3(const RenderGraphProperties* camera, const char* key, const float3& fallback)
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

Result<> createMeshShader(Device& device, std::unique_ptr<ShaderModule>& outShader, std::string& log)
{
    ShaderCompileResult meshCompile;
    const char* capabilities[] = {"spvMeshShadingEXT"};
    Result<> result = compileSlangShaderToSpirv(SlangShaderDesc{
        .moduleName = kMeshletStreamShaderModuleName,
        .entryPointName = kMeshletStreamMeshEntryPoint,
        .searchPath = kMeshletStreamShaderSearchPath,
        .capabilities = {capabilities, static_cast<uint32_t>(std::size(capabilities))},
    }, meshCompile.diagnostics).transform([&](auto value) { meshCompile = std::move(value); });
    if (!result) {
        log += "compileSlangShaderToSpirv(GPUDrivenStreamAsset.mesh) returned ";
        log += resultToString(result);
        if (!meshCompile.diagnostics.empty()) {
            log += ": ";
            log += meshCompile.diagnostics;
        }
        log += '\n';
        return result;
    }

    result = device.createShaderModule(ShaderModuleDesc{
        .spirv = meshCompile.spirv,
        .debugName = "GPUDrivenStreamAsset.mesh",
    }).transform([&](auto rhiValue) { outShader = std::move(rhiValue); });
    if (!result || outShader == nullptr) {
        log += resultMessage("createShaderModule(GPUDrivenStreamAsset mesh)", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

Result<> createStreamShader(
    Device& device,
    const char* entryPoint,
    std::unique_ptr<ShaderModule>& outShader,
    std::string& log)
{
    const bool composite = std::string_view(entryPoint) == kMeshletStreamCompositeVertexEntryPoint ||
        std::string_view(entryPoint) == kMeshletStreamCompositeFragmentEntryPoint;
    const char* module = composite ? kMeshletStreamCompositeShaderModuleName : kMeshletStreamShaderModuleName;
    ShaderCompileResult compileResult;
    Result<> result = compileSlangShader(
        module,
        entryPoint,
        compileResult,
        log);
    if (!result) {
        return result;
    }
    const std::string shaderDebugName =
        std::string(module) + "." + entryPoint;
    result = device.createShaderModule(ShaderModuleDesc{
        .spirv = compileResult.spirv,
        .debugName = shaderDebugName.c_str(),
    }).transform([&](auto rhiValue) { outShader = std::move(rhiValue); });
    if (!result || outShader == nullptr) {
        log += resultMessage(
            std::string("createShaderModule(GPUDrivenStreamAsset ") + entryPoint + ")",
            result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

struct GPUDrivenStreamAssetRetiredFrameResources {
    std::unique_ptr<Buffer> deferredColorBuffer;
};

} // namespace

class GPUDrivenStreamAssetPass final : public UnsafePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.geometry = SceneStreamKind::Asset};
    }
    void describeSceneView(const RenderGraphExecutionContext& context, MeshletStreamFrameDesc& view) const override
    {
        view = frameDescFromContext(context);
    }

    RenderGraphSceneDependency sceneDependency() const override
    {
        return {boolProperty(properties(), "streamAssetOnly", false)
            ? RenderGraphSceneSource::None : RenderGraphSceneSource::World};
    }

    ~GPUDrivenStreamAssetPass() override
    {
        releaseRasterHandles();
        releaseGPUSceneSourceLease();
        if (gpuSceneSubsystem_ != nullptr && gpuSceneView_.valid()) {
            gpuSceneSubsystem_->destroyView(gpuSceneView_);
        }
    }

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{
            GPUSceneSubsystem::kSubsystemId,
            StreamerSubsystem::kSubsystemId,
        };
        return required;
    }

    std::vector<std::string> debugCheckpoints() const override
    {
        return rtasVisualization_ ? std::vector<std::string>{"AfterTraversal", "AfterPass"}
            : std::vector<std::string>{"AfterTraversal", "AfterEarlyCull", "AfterLateCull", "AfterPass"};
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureOutput("accelerationStructure", "Stream TLAS for ray-query consumers")
            .buildWrite().stageAccess(RenderGraphResourceAccess::AccelerationStructureShaderRead).setOptional();
        RenderGraphField& color = reflection.addTextureOutput(
            "color",
            "Meshlet streamasset deferred color");
        const bool rtasVisualization = boolProperty(properties(), "rtasVisualization", false);
        if (rtasVisualization) {
            color.texture2D().storageReadWrite().format = Format::RGBA8Unorm;
            color.stageAccess(RenderGraphResourceAccess::TextureStorageWrite);
            color.transient(RenderGraphInitialization::FullOverwrite);
        } else {
            color.texture2D().colorWrite().transient(RenderGraphInitialization::Clear);
        }
        RenderGraphField& visibility = reflection.addTextureOutput(
            "visibility",
            "Stream mesh shader visibility IDs");
        visibility.colorWrite();
        visibility.format = Format::R32Uint;
        visibility.stageAccess(RenderGraphResourceAccess::TextureSampleRead);
        if (!rtasVisualization) { visibility.transient(RenderGraphInitialization::Clear); }
        RenderGraphField& depth = reflection.addTextureOutput(
            "depth",
            "Meshlet streamasset visibility depth and HZB source");
        depth.texture2D().depthStencilWrite();
        depth.stageAccess(RenderGraphResourceAccess::TextureSampleRead);
        if (!rtasVisualization) { depth.transient(RenderGraphInitialization::Clear); }
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        auto prefetch = runtimeBoolSetting("predictivePrefetch", "Predictive Geometry Prefetch", true);
        prefetch.rebuildGraph = true;
        auto retention = runtimeBoolSetting("adaptivePageRetention", "Adaptive Page Retention", true);
        retention.rebuildGraph = true;
        auto telemetry = runtimeBoolSetting("enableLodTransitionTelemetry", "LOD Transition Diagnostics (Extra GPU Memory)", false);
        telemetry.rebuildGraph = true;
        return {
            prefetch,
            {.key = "AsyncComputePreferred", .label = "Async RTAS Build",
                .type = RenderGraphRuntimeSettingType::Bool, .defaultValue = true, .rebuildGraph = true},
            retention,
            telemetry,
            runtimeBoolSetting("autoLod", "Auto Meshlet LOD", autoLodProperty(properties())),
            runtimeFloatSetting("lodPixelError", "LOD Error (display px)", 1.5f, 0.05f, 16.0f),
            runtimeFloatSetting("lodBias", "LOD Bias", 0.0f, -4.0f, 4.0f),
            runtimeIntSetting("lodLevel", "Manual LOD (Auto Off)", static_cast<int32_t>(selectedLodProperty(properties())), 0, 31),
            runtimeBoolSetting("instanceFrustumCull", "Instance Frustum Cull", true),
            runtimeBoolSetting("instanceHzbCull", "Instance HZB Cull", true),
            runtimeBoolSetting("clusterFrustumCull", "Cluster Sphere / Frustum Cull", true),
            runtimeBoolSetting("clusterNormalConeCull", "Cluster Normal Cone Cull", true),
            runtimeEnumSetting(
                "debugColorMode",
                "Color",
                "shaded",
                {
                    {"Shaded", "shaded"},
                    {"Page", "page"},
                    {"LOD", "lod"},
                    {"Primitive", "primitive"},
                    {"Instance", "instance"},
                    {"Meshlet / Cluster", "meshlet"},
                }),
            runtimeEnumSetting(
                "rtasGranularity",
                "RTAS Color",
                "instance",
                {
                    {"Instance", "instance"},
                    {"Primitive", "primitive"},
                    {"Cluster ID", "cluster-id"},
                }),
        };
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        log.clear();
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        // Slang emits the Geometry capability for fragment SV_PrimitiveID,
        // including IDs supplied by a mesh shader primitive output.
        if (!context.device->capabilities().meshShader ||
            !context.device->capabilities().geometryShader ||
            !context.device->capabilities().bindlessDescriptorHeap) {
            log = "GPUDrivenStreamAssetPass requires meshShader, geometryShader, and bindlessDescriptorHeap capabilities";
            return makeError(Error::Unsupported);
        }

        GPUSceneSubsystem* gpuSceneSubsystem = context.subsystem<GPUSceneSubsystem>();
        if (gpuSceneSubsystem == nullptr) {
            log = "GPUDrivenStreamAssetPass requires GPUSceneSubsystem";
            return makeError(Error::InvalidArgument);
        }
        if (gpuSceneSubsystem_ != gpuSceneSubsystem) {
            releaseGPUSceneSourceLease();
            if (gpuSceneSubsystem_ != nullptr && gpuSceneView_.valid()) {
                gpuSceneSubsystem_->destroyView(gpuSceneView_);
            }
            gpuSceneSubsystem_ = gpuSceneSubsystem;
            gpuSceneView_ = {};
        }

        const bool streamAssetOnly = boolProperty(properties(), "streamAssetOnly", false);
        const scene::Scene* runtimeScene = streamAssetOnly ? nullptr : runtimeSceneForPath(
            context.runtimeScene, scenePathFromProperties(properties()));

        if (gpuSceneSource_ != runtimeScene) {
            releaseGPUSceneSourceLease();
            gpuSceneSource_ = runtimeScene;
            if (gpuSceneSource_ != nullptr) {
                Result<> leaseResult = gpuSceneSubsystem_->acquireSourceOverride(gpuSceneSource_, log).transform([&](auto value) { gpuSceneSourceToken_ = std::move(value); });
                if (!leaseResult) {
                    gpuSceneSource_ = nullptr;
                    return leaseResult;
                }
            }
        }

        if (!context.preparedScene || !context.preparedScene->geometry) { return makeError(Error::InvalidArgument); }
        const auto preparedStream = context.preparedScene->geometry;
        const uint64_t sourceIdentity = runtimeScene != nullptr ? runtimeScene->resourceIdentity() : 0;
        const uint64_t sourceContentRevision = runtimeScene != nullptr ? runtimeScene->contentRevision() : 0;
        const bool rtasVisualization = boolProperty(properties(), "rtasVisualization", false);
        if (rtasVisualization) {
            if (!boolProperty(properties(), "enableClusterRtx", false)) {
                log = "GPUDrivenStreamAssetPass RTAS visualization requires enableClusterRtx=true";
                return makeError(Error::InvalidArgument);
            }
            if (!context.device->capabilities().rayQuery ||
                !context.device->capabilities().clusterAccelerationStructure) {
                log = "GPUDrivenStreamAssetPass RTAS visualization requires rayQuery and clusterAccelerationStructure";
                return makeError(Error::Unsupported);
            }
        }

        // Scene-binding refresh also recompiles for transform/visibility edits.
        // Those edits update instance data; they must not evict resident pages,
        // restart page uploads, or allocate another set of bindless handles.
        if (compiled_ && device_ == context.device && gpuSceneView_.valid() &&
            frameSlotCount_ == std::max(gpuSceneSubsystem_->frameSlotCount(), 1u) &&
            (streamRuntime_ && streamRuntime_->ready()) && streamRuntime_ == preparedStream &&
            compiledSourceIdentity_ == sourceIdentity &&
            compiledSourceContentRevision_ == sourceContentRevision &&
            compiledStreamAssetOnly_ == streamAssetOnly &&
            compiledDebugReadback_ == context.debugReadback &&
            compiledColorFormat_ == context.defaultFormat &&
            rtasVisualization_ == rtasVisualization) {
            Result<> result;

            result = ensureFrameResources(context.width, context.height, context.subsystems());
            if (!result) { return result; }
            rtasVisualization_ = rtasVisualization;
            hzbValid_ = false;
            for (uint32_t slot = 0; slot < frameSlotCount_; ++slot) {
                (void)gpuSceneSubsystem_->markViewHzbValid(gpuSceneView_, slot, false);
            }
            return {};
        }

        compiled_ = false;
        rayQueryProgram_.clear();
        Result<> result = context.device->createPipelineCache(PipelineCacheDesc{
            .filePath = PROJECT_SOURCE_DIR "/.cache/pso/GPUDrivenStreamAssetPass.pso"}).transform([&](auto rhiValue) { pipelineCache_ = std::move(rhiValue); });
        if (!result || pipelineCache_ == nullptr) {
            log += resultMessage("createPipelineCache(GPUDrivenStreamAssetPass)", result);
            return result ? makeError(Error::Failure) : result;
        }
        releaseRasterHandles();
        streamRuntime_ = preparedStream;
        rtasVisualization_ = rtasVisualization;

        result = createMeshShader(*context.device, meshShader_, log);
        if (!result) {
            return result;
        }

        result = createStreamShader(
            *context.device,
            kMeshletStreamFragmentEntryPoint,
            fragmentShader_,
            log);
        if (!result) {
            return result;
        }
        result = createStreamShader(
            *context.device,
            kMeshletStreamCompositeVertexEntryPoint,
            compositeVertexShader_,
            log);
        if (!result) {
            return result;
        }
        result = createStreamShader(
            *context.device,
            kMeshletStreamCompositeFragmentEntryPoint,
            compositeFragmentShader_,
            log);
        if (!result) {
            return result;
        }
        for (uint32_t reversedZ = 0; reversedZ < visibilityPipelines_.size(); ++reversedZ) {
            result = context.device->createGraphicsPipeline(GraphicsPipelineDesc{
                .meshShader = {meshShader_.get()},
                .fragmentShader = {fragmentShader_.get()},
                .colorFormat = Format::R32Uint,
                .depthStencilFormat = Format::D32Sfloat,
                .depthStencil = DepthStencilState{
                        .depthTestEnable = true,
                        .depthWriteEnable = true,
                        .depthCompareOp = depthCompareOp(reversedZ != 0u),
                    },
                .usesBindlessHeap = true,
            }).transform([&](auto rhiValue) { visibilityPipelines_[reversedZ] = std::move(rhiValue); });
            if (!result || visibilityPipelines_[reversedZ] == nullptr) {
                log += resultMessage("createGraphicsPipeline(GPUDrivenStreamAsset visibility)", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
        }

        ShaderCompileResult deferredCompile;
        result = compileSlangShader(kMeshletStreamShaderModuleName, kMeshletStreamDeferredEntryPoint, deferredCompile, log);
        if (!result) { return result; }
        result = deferredKernel_.initialize(*context.device, {.spirv = deferredCompile.spirv,
            .parameters = parameterAbi<StreamDeferredParameters>(kStreamDeferredABI, ParameterTransport::InlinePush),
            .debugName = "Stream deferred"}, log);
        if (!result) { return result; }
        result = context.device->createGraphicsPipeline(GraphicsPipelineDesc{
            .vertexShader = {compositeVertexShader_.get()},
            .fragmentShader = {compositeFragmentShader_.get()},
            .colorFormat = context.defaultFormat,
            .usesBindlessHeap = true,
        }).transform([&](auto rhiValue) { compositePipeline_ = std::move(rhiValue); });
        if (!result || compositePipeline_ == nullptr) {
            log += resultMessage("createGraphicsPipeline(GPUDrivenStreamAsset composite)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        auto createCullKernel = [&](const char* entry, ComputeKernel& kernel) -> Result<> {
            ShaderCompileResult compiled;
            auto compiledResult = compileSlangShader(kMeshletStreamShaderModuleName, entry, compiled, log);
            if (!compiledResult) { return compiledResult; }
            return kernel.initialize(*context.device, {.spirv = compiled.spirv,
                .parameters = parameterAbi<StreamInstanceCullParameters>(kStreamInstanceCullABI, ParameterTransport::InlinePush),
                .debugName = entry}, log);
        };
        result = createCullKernel(kMeshletStreamCullResetEntryPoint, cullResetKernel_);
        if (!result) { return result; }
        result = createCullKernel(kMeshletStreamInstanceCullEntryPoint, instanceCullKernel_);
        if (!result) { return result; }
        ShaderCompileResult hzbCompile;
        result = compileSlangShader("Features/GPUDriven/HZB", kMeshletStreamHZBEntryPoint, hzbCompile, log);
        if (!result) { return result; }
        result = hzbKernel_.initialize(*context.device, {.spirv = hzbCompile.spirv,
            .parameters = parameterAbi<HZBParameters>(kHZBABI, ParameterTransport::InlinePush),
            .debugName = "Stream HZB"}, log);
        if (!result) {
            return result;
        }

        frameWidth_ = std::max(context.width, 1u);
        frameHeight_ = std::max(context.height, 1u);
        hzbMipCount_ = computeHzbMipCount(frameWidth_, frameHeight_);
        hzbElementCount_ = computeHzbElementCount(
            frameWidth_,
            frameHeight_,
            hzbMipCount_);

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
            auto view = gpuSceneSubsystem_->createView(GPUSceneViewDesc{
                .frameSlotCount = requestedFrameSlotCount,
            }, log);
            if (!view) { return makeError(view.error()); }
            gpuSceneView_ = *view;
        }
        frameSlotCount_ = requestedFrameSlotCount;
        instanceCapacity_ = std::max<uint32_t>(
            static_cast<uint32_t>(streamRuntime_->asset().instances().size()),
            1u);
        for (const scene::MeshletStreamInstanceInfo& instance :
             streamRuntime_->asset().instances()) {
            if (instance.renderNodeIndex != std::numeric_limits<uint32_t>::max()) {
                instanceCapacity_ = std::max(
                    instanceCapacity_,
                    instance.renderNodeIndex + 1u);
            }
        }
        if (runtimeScene != nullptr) {
            instanceCapacity_ = std::max<uint32_t>(
                instanceCapacity_,
                static_cast<uint32_t>(runtimeScene->renderNodes().size()));
        }
        std::array<uint32_t, kGPUSceneRasterDrawBucketCount> phaseCapacities{};
        phaseCapacities.fill(1u);
        result = gpuSceneSubsystem_->ensureViewGpuResources(
            gpuSceneView_,
            GPUSceneViewDesc{
                .frameSlotCount = requestedFrameSlotCount,
                .instanceCapacity = instanceCapacity_,
                .visibleMeshletCapacity = phaseCapacities,
                .hzbWidth = frameWidth_,
                .hzbHeight = frameHeight_,
                .hzbMipCount = hzbMipCount_,
                .hzbElementCount = hzbElementCount_,
            },
            log);
        if (!result) {
            return result;
        }

        result = context.device->createBuffer(BufferDesc{
                .size = static_cast<uint64_t>(frameWidth_) * frameHeight_ * sizeof(uint32_t),
                .structureStride = sizeof(uint32_t),
                .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::Device,
            }).transform([&](auto rhiValue) { deferredColorBuffer_ = std::move(rhiValue); });
        if (!result || deferredColorBuffer_ == nullptr) {
            log += resultMessage("createBuffer(GPUDrivenStreamAsset deferred color)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = streamRuntime_->updateRasterBindings(MeshletStreamGPURasterBindings{
            .instanceVisibilityBuffer = {uint64_t(instanceVisibilityHandle_.shaderValue())},
            .hzbBuffer0 = {uint64_t(hzbHandles_[0].shaderValue())},
            .hzbBuffer1 = {uint64_t(hzbHandles_[1].shaderValue())},
            .hzbMipCount = hzbMipCount_,
            .hzbValid = 0u,
            .cullingFlags = cullingFlagsFromProperties(properties()),
            .width = frameWidth_,
            .height = frameHeight_,
        });
        if (!result) {
            log = "GPUDrivenStreamAssetPass failed to publish raster bindings";
            return result;
        }
        if (rtasVisualization_) {
            result = initializeRayQuery(*context.device, log);
            if (!result) {
                return result;
            }
        }

        device_ = context.device;
        compiledSourceIdentity_ = sourceIdentity;
        compiledSourceContentRevision_ = sourceContentRevision;
        compiledStreamAssetOnly_ = streamAssetOnly;
        compiledDebugReadback_ = context.debugReadback;
        compiledColorFormat_ = context.defaultFormat;
        const Result<> saveResult = pipelineCache_->save();
        const PipelineCacheStats cacheStats = pipelineCache_->stats();
        spdlog::info("[GPUDrivenStreamAssetPass] PSO cache hits={} misses={} stored={} bytes={}",
            cacheStats.hitCount, cacheStats.missCount, cacheStats.storedPsoCount, cacheStats.backendDataSize);
        if (!saveResult) { log += "Warning: GPUDrivenStreamAssetPass failed to save PSO cache\n"; }
        compiled_ = true;
        return {};
    }

    Result<> prepareExecution(RenderGraphExecutionContext& context) override
    {
        if (streamRuntime_) {
            if (auto* frame = context.commandBuffer().frameContext()) { frame->retain(streamRuntime_); }
        }
        GPUSceneSubsystem* gpuSceneSubsystem = context.subsystem<GPUSceneSubsystem>();
        if (gpuSceneSubsystem == nullptr ||
            gpuSceneSubsystem != gpuSceneSubsystem_ ||
            !gpuSceneView_.valid()) {
            return makeError(Error::InvalidArgument);
        }
        Result<> result;
        TextureHandle color = context.outputTexture("color");
        TextureHandle visibility = context.outputTexture("visibility");
        TextureHandle depth = context.outputTexture("depth");
        if (!color.valid() ||
            !visibility.valid() ||
            !depth.valid() ||
            context.streamer() == nullptr ||
            !(streamRuntime_ && streamRuntime_->ready()) ||
            streamRuntime_->bindlessHeap() == nullptr ||
            visibilityPipelines_[0] == nullptr || visibilityPipelines_[1] == nullptr ||
            !deferredKernel_.valid() ||
            compositePipeline_ == nullptr ||
            !cullResetKernel_.valid() ||
            !instanceCullKernel_.valid() ||
            !hzbKernel_.valid() ||
            (rtasVisualization_ && !rayQueryProgram_.valid())) {
            return makeError(Error::InvalidArgument);
        }

        result = ensureFrameResources(
            context.width(),
            context.height(),
            context.subsystems());
        if (!result || deferredColorBuffer_ == nullptr) {
            return result ? makeError(Error::Failure) : result;
        }

        const MeshletStreamFrameDesc frame = frameDescFromContext(context);
        result = streamRuntime_->resourceRegistry()->sampledImage(*visibility.view(), ResourceState::ShaderRead).transform([&](auto value) { visibilityImageHandle_ = std::move(value); });
        if (!result) {
            return result;
        }
        result = streamRuntime_->resourceRegistry()->sampledImage(*depth.view(), ResourceState::ShaderRead).transform([&](auto value) { depthImageHandle_ = std::move(value); });
        if (!result) {
            return result;
        }
        if (!rtasVisualization_) {
            bool cameraCut = false;
            if (HistoryResourceManager* historyResources = context.historyResources()) {
                const uint64_t invalidationRevision =
                    historyResources->invalidationRevision();
                cameraCut = observedHistoryInvalidationRevision_ != 0 &&
                    observedHistoryInvalidationRevision_ != invalidationRevision;
                observedHistoryInvalidationRevision_ = invalidationRevision;
            }
            result = prepareGPUSceneView(*gpuSceneSubsystem, cameraCut, frame);
            if (!result) {
                return result;
            }
            result = bindGPUSceneViewResources(*gpuSceneSubsystem, depth);
            if (!result) {
                return result;
            }
            std::string gpuSceneLog;
            result = gpuSceneSubsystem->recordInitialize(
                context.commandBuffer(),
                gpuSceneView_,
                activeFrameSlot_,
                gpuSceneLog);
            if (!result) {
                spdlog::error("[GPUDrivenStreamAssetPass] {}", gpuSceneLog);
                return result;
            }

            // Empty candidate sets still dispatch so stale cell contents cannot
            // survive into a later lighting consumer for this view/frame slot.
            result = gpuSceneSubsystem->recordLightGrid(
                context.commandBuffer(),
                gpuSceneView_,
                activeFrameSlot_,
                lightGridDesc_,
                gpuSceneLog);
            if (!result) {
                spdlog::error("[GPUDrivenStreamAssetPass] {}", gpuSceneLog);
                return result;
            }

        }
        return {};
    }

    void sceneTraversalCheckpoint(RenderGraphExecutionContext& context, std::string_view point) const override
    {
        if (point == "AfterTraversal") {
            gpuDrivenDebugCheckpoint(context, point, gpuSceneSubsystem_, gpuSceneView_, activeFrameSlot_,
                streamRuntime_.get(), UINT32_MAX);
        }
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto* gpuSceneSubsystem = context.subsystem<GPUSceneSubsystem>();
        const auto color = context.outputTexture("color");
        const auto visibility = context.outputTexture("visibility");
        const auto depth = context.outputTexture("depth");
        const auto frame = frameDescFromContext(context);
        Result<> result;
        auto& registry = *streamRuntime_->resourceRegistry();
        for (const auto& lease : {visibilityImageHandle_, depthImageHandle_,
             instanceVisibilityHandle_, visibleInstanceIdsHandle_, visibleInstanceCounterHandle_,
             hzbHandles_[0], hzbHandles_[1]}) {
            if (lease.valid()) {
                result = registry.retain(context.commandBuffer(), lease);
                if (!result) { return result; }
            }
        }
        result = registry.bind(context.commandBuffer());
        if (!result) { return result; }

        using Access = RenderGraphResourceAccess;
        if (rtasVisualization_) {
            if (streamRuntime_->topLevelBuildPending()) {
                result = context.parallelCompute([&](CommandBuffer& commands) {
                    auto profile = context.profileScope(commands, "Async TLAS build");
                    return streamRuntime_->cmdBuildTopLevelAccelerationStructure(commands);
                }, [](CommandBuffer&) -> Result<> { return {}; });
                if (!result) { return result; }
            }
            result = context.publishAccelerationStructure("accelerationStructure", streamRuntime_->accelerationStructure());
            if (!result) { return result; }
            const RenderGraphStageUse rayQueryUses[] = {{"color", Access::TextureStorageWrite},
                {"accelerationStructure", Access::AccelerationStructureShaderRead}};
            const RenderGraphStage stages[] = {{"Ray query", rayQueryUses,
                [&](CommandBuffer&) { return drawRayQuery(context, color, frame); }}};
            result = context.executeStages(stages);
        } else {
            const RenderGraphStageUse rasterUses[] = {
                {"visibility", Access::TextureColorWrite},
                {"depth", Access::TextureDepthStencilWrite},
            };
            const RenderGraphStageUse hzbUses[] = {{"depth", Access::TextureSampleRead}};
            const RenderGraphStageUse deferredUses[] = {
                {"visibility", Access::TextureSampleRead},
                {"deferredColor", Access::BufferStorageWrite},
            };
            const RenderGraphStageUse compositeUses[] = {
                {"deferredColor", Access::BufferShaderRead},
                {"color", Access::TextureColorWrite},
            };
            // GPUScene and streaming recorders retain their private-resource
            // synchronization contracts. This sequence owns graph image layouts
            // and the pass-owned deferred buffer across their opaque operations.
            const auto cull = [&](CommandBuffer& commands, GPUSceneCullPhase phase) -> Result<> {
                auto cullResult = dispatchInstanceCull(commands, phase);
                if (!cullResult) { return cullResult; }
                gpuDrivenDebugCheckpoint(context,
                    phase == GPUSceneCullPhase::Early ? "AfterEarlyCull" : "AfterLateCull",
                    gpuSceneSubsystem, gpuSceneView_, activeFrameSlot_, streamRuntime_.get(),
                    phase == GPUSceneCullPhase::Early ? 0u : 1u);
                return streamRuntime_->cmdPrepareVisibility(commands);
            };
            const RenderGraphStage stages[] = {
                {"Early Instance cull", {}, [&](CommandBuffer& commands) {
                    return streamRuntime_->topLevelBuildPending()
                        ? context.parallelCompute([&](CommandBuffer& buildCommands) {
                            auto profile = context.profileScope(buildCommands, "Async TLAS build");
                            return streamRuntime_->cmdBuildTopLevelAccelerationStructure(buildCommands);
                        }, [&](CommandBuffer& cullCommands) { return cull(cullCommands, GPUSceneCullPhase::Early); })
                        : cull(commands, GPUSceneCullPhase::Early);
                }, RenderGraphPassKind::Unsafe, true},
                {"Early Hardware raster", rasterUses, [&](CommandBuffer&) {
                    return draw(context, *visibility.view(), depth, GPUSceneCullPhase::Early,
                        LoadOp::Clear, frame.camera.reversedZ);
                }, RenderGraphPassKind::Raster},
                {"Early HZB", hzbUses, [&](CommandBuffer& commands) { return buildHzb(commands, depth.view(), frame.camera.reversedZ); }},
                {"Late Instance cull", {}, [&](CommandBuffer& commands) {
                    return cull(commands, GPUSceneCullPhase::Late);
                }},
                {"Late Hardware raster", rasterUses, [&](CommandBuffer&) {
                    return draw(context, *visibility.view(), depth, GPUSceneCullPhase::Late,
                        LoadOp::Load, frame.camera.reversedZ);
                }, RenderGraphPassKind::Raster},
                {"Late HZB", hzbUses, [&](CommandBuffer& commands) { return buildHzb(commands, depth.view(), frame.camera.reversedZ); }},
                {"Deferred shading", deferredUses, [&](CommandBuffer&) { return dispatchDeferred(context); }},
                {"Composite", compositeUses, [&](CommandBuffer&) {
                    return drawComposite(context, color);
                }, RenderGraphPassKind::Raster},
            };
            auto deferredColor = deferredColorBuffer_->slice();
            if (!deferredColor) { return makeError(deferredColor.error()); }
            const RenderGraphBufferImport imports[] = {
                {"deferredColor", *deferredColor, Access::BufferStorageReadWrite},
            };
            result = context.executeStages(stages, imports);
            if (result) {
                hzbValid_ = true;
                if (!gpuSceneSubsystem->markViewHzbValid(
                        gpuSceneView_,
                        activeFrameSlot_,
                        true)) {
                    result = makeError(Error::Failure);
                }
            }
        }
        if (!result) {
            return result;
        }
        if (result) { gpuDrivenDebugCheckpoint(context, "AfterPass", gpuSceneSubsystem, gpuSceneView_, activeFrameSlot_, streamRuntime_.get(), rtasVisualization_ ? UINT32_MAX : 1); }
        return context.publishAccelerationStructure("accelerationStructure",
            streamRuntime_->tlasReady() ? streamRuntime_->accelerationStructure() : nullptr);
    }

private:
    void releaseRasterHandles()
    {
        visibilityImageHandle_ = {};
        depthImageHandle_ = {};
        instanceVisibilityHandle_ = {};
        visibleInstanceIdsHandle_ = {};
        visibleInstanceCounterHandle_ = {};
        hzbHandles_ = {};
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

    static uint64_t computeHzbElementCount(
        uint32_t width,
        uint32_t height,
        uint32_t mipCount)
    {
        uint64_t elementCount = 0;
        for (uint32_t mipLevel = 0; mipLevel < mipCount; ++mipLevel) {
            elementCount += static_cast<uint64_t>(width) * height;
            width = std::max(1u, (width + 1u) / 2u);
            height = std::max(1u, (height + 1u) / 2u);
        }
        return elementCount;
    }

    Result<> ensureFrameResources(
        uint32_t width,
        uint32_t height,
        RenderSubsystemHost* subsystemHost)
    {
        width = std::max(width, 1u);
        height = std::max(height, 1u);
        if (frameWidth_ == width && frameHeight_ == height) {
            return {};
        }
        if (device_ == nullptr ||
            gpuSceneSubsystem_ == nullptr ||
            !gpuSceneView_.valid() ||
            subsystemHost == nullptr ||
            streamRuntime_->bindlessHeap() == nullptr ||
            deferredColorBuffer_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const uint32_t mipCount = computeHzbMipCount(width, height);
        const uint64_t elementCount = computeHzbElementCount(width, height, mipCount);
        std::unique_ptr<Buffer> resizedDeferredColorBuffer;
        Result<> result = device_->createBuffer(BufferDesc{
                .size = static_cast<uint64_t>(width) * height * sizeof(uint32_t),
                .structureStride = sizeof(uint32_t),
                .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::Device,
            }).transform([&](auto rhiValue) { resizedDeferredColorBuffer = std::move(rhiValue); });
        if (!result || resizedDeferredColorBuffer == nullptr) {
            spdlog::error(
                "[GPUDrivenStreamAssetPass] {}",
                resultMessage("createBuffer(resized deferred color)", result));
            return result ? makeError(Error::Failure) : result;
        }

        std::array<uint32_t, kGPUSceneRasterDrawBucketCount> phaseCapacities{};
        phaseCapacities.fill(1u);
        std::string log;
        result = gpuSceneSubsystem_->ensureViewGpuResources(
            gpuSceneView_,
            GPUSceneViewDesc{
                .frameSlotCount = frameSlotCount_,
                .instanceCapacity = instanceCapacity_,
                .visibleMeshletCapacity = phaseCapacities,
                .hzbWidth = width,
                .hzbHeight = height,
                .hzbMipCount = mipCount,
                .hzbElementCount = elementCount,
            },
            log);
        if (!result) {
            spdlog::error("[GPUDrivenStreamAssetPass] {}", log);
            return result;
        }

        auto retired = std::make_shared<GPUDrivenStreamAssetRetiredFrameResources>();
        retired->deferredColorBuffer = std::move(deferredColorBuffer_);
        deferredColorBuffer_ = std::move(resizedDeferredColorBuffer);
        subsystemHost->retire(std::static_pointer_cast<void>(retired));

        spdlog::info(
            "[GPUDrivenStreamAssetPass] Resized frame resources {}x{} -> {}x{}",
            frameWidth_,
            frameHeight_,
            width,
            height);
        frameWidth_ = width;
        frameHeight_ = height;
        hzbMipCount_ = mipCount;
        hzbElementCount_ = elementCount;
        hzbValid_ = false;
        return {};
    }

    Result<> prepareGPUSceneView(
        GPUSceneSubsystem& subsystem,
        bool cameraCut,
        const MeshletStreamFrameDesc& frame)
    {
        activeFrameSlot_ = subsystem.currentFrameSlot();
        if (activeFrameSlot_ >= frameSlotCount_) {
            return makeError(Error::InvalidArgument);
        }
        GPUSceneViewPrepareInfo prepareInfo{
            .width = frameWidth_,
            .height = frameHeight_,
            .cameraCut = cameraCut,
            .freezeCullingCamera = false,
        };
        // Consume the same camera and projection uploaded for LOD selection.
        if (!std::isfinite(frame.camera.fovDegrees) ||
            !std::isfinite(frame.camera.znear) ||
            !std::isfinite(frame.camera.zfar) ||
            !std::isfinite(frame.camera.orthoHeight)) {
            spdlog::error("[GPUDrivenStreamAssetPass] Light grid camera has non-finite projection parameters");
            return makeError(Error::InvalidArgument);
        }
        lightGridDesc_ = ClusterLightGridDesc{
            .width = frameWidth_,
            .height = frameHeight_,
            .eye = frame.camera.eye,
            .center = frame.camera.center,
            .up = frame.camera.up,
            .aspect = std::max(static_cast<float>(std::max(frame.width, 1u)) /
                static_cast<float>(std::max(frame.height, 1u)), 0.001f),
            .fovRadians = std::clamp(frame.camera.fovDegrees * 0.017453292519943295f,
                0.017453292f, 3.12413936f),
            .zNear = std::max(frame.camera.znear, 0.0001f),
            .zFar = std::max(frame.camera.zfar, std::max(frame.camera.znear, 0.0001f) + 0.0001f),
            .orthoHeight = frame.camera.orthographic ? std::max(frame.camera.orthoHeight, 0.0001f) : 0.0f,
        };
        if (boolProperty(properties(), "instanceFrustumCull", true)) {
            prepareInfo.lightFrustumPlanes = gpuSceneLightFrustumPlanes(
                lightGridDesc_.eye, lightGridDesc_.center, lightGridDesc_.up,
                lightGridDesc_.aspect, lightGridDesc_.fovRadians,
                lightGridDesc_.zNear, lightGridDesc_.zFar, lightGridDesc_.orthoHeight);
        }
        if (!subsystem.prepareView(
                gpuSceneView_,
                activeFrameSlot_,
                prepareInfo)) {
            return makeError(Error::InvalidArgument);
        }
        const GPUSceneVisibleDrawSet* visible = subsystem.visibleDrawSet(
            gpuSceneView_,
            activeFrameSlot_);
        if (visible == nullptr) {
            return makeError(Error::Failure);
        }
        hzbValid_ = visible->stats.hzbValid && !cameraCut;
        std::string log;
        Result<> result = subsystem.publishViewGpuResources(
            gpuSceneView_,
            activeFrameSlot_,
            streamRuntime_->frameIndex() & 1u,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenStreamAssetPass] {}", log);
        }
        return result;
    }

    Result<> bindGPUSceneViewResources(
        GPUSceneSubsystem& subsystem,
        TextureHandle depth)
    {
        GPUSceneViewGPUResourcesView resources;
        if (!depth.valid() ||
            !subsystem.viewGpuResources(
                gpuSceneView_,
                activeFrameSlot_,
                resources) ||
            resources.instanceVisibilityStates.buffer == nullptr ||
            resources.visibleInstanceIds.buffer == nullptr ||
            resources.visibleInstanceCounter.buffer == nullptr ||
            resources.hzbHistory[0].buffer == nullptr ||
            resources.hzbHistory[1].buffer == nullptr ||
            streamRuntime_->bindlessHeap() == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        ResourceRegistry& heap = *streamRuntime_->resourceRegistry();
        Result<> result = heap.storageBuffer(*resources.instanceVisibilityStates.buffer).transform([&](auto value) { instanceVisibilityHandle_ = std::move(value); });
        if (result) {
            result = heap.storageBuffer(*resources.visibleInstanceIds.buffer).transform([&](auto value) { visibleInstanceIdsHandle_ = std::move(value); });
        }
        if (result) {
            result = heap.storageBuffer(*resources.visibleInstanceCounter.buffer).transform([&](auto value) { visibleInstanceCounterHandle_ = std::move(value); });
        }
        for (uint32_t historyIndex = 0;
             historyIndex < hzbHandles_.size() && result;
             ++historyIndex) {
            result = heap.storageBuffer(*resources.hzbHistory[historyIndex].buffer).transform([&](auto value) { hzbHandles_[historyIndex] = std::move(value); });
        }
        if (!result) {
            return result;
        }

        return streamRuntime_->updateRasterBindings(MeshletStreamGPURasterBindings{
            .instanceVisibilityBuffer = {uint64_t(instanceVisibilityHandle_.shaderValue())},
            .hzbBuffer0 = {uint64_t(hzbHandles_[0].shaderValue())},
            .hzbBuffer1 = {uint64_t(hzbHandles_[1].shaderValue())},
            .hzbMipCount = hzbMipCount_,
            .hzbValid = hzbValid_ ? 1u : 0u,
            .cullingFlags = cullingFlagsFromProperties(properties()),
            .width = frameWidth_,
            .height = frameHeight_,
        });
    }

    Result<> dispatchInstanceCull(
        CommandBuffer& commandBuffer,
        GPUSceneCullPhase phase)
    {
        if (gpuSceneSubsystem_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        const auto* visible = gpuSceneSubsystem_->visibleDrawSet(gpuSceneView_, activeFrameSlot_);
        if (visible == nullptr) { return makeError(Error::InvalidArgument); }
        const auto& gpu = visible->gpu;
        const uint32_t phaseIndex = phase == GPUSceneCullPhase::Early ? 0u : 1u;
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, commandBuffer.frameContext());
        auto encoded = streamRuntime_->encodeInstanceCull(writer, gpu.instanceVisibilityStates.buffer,
            gpu.visibleInstanceIds.buffer, gpu.visibleInstanceCounter.buffer,
            gpu.hzb.history[gpu.hzb.writeIndex ^ (phaseIndex == 0u ? 1u : 0u)].buffer, phaseIndex);
        if (!encoded) { return makeError(encoded.error()); }
        auto reset = cullResetKernel_.prepareDispatch(*encoded, 1);
        if (!reset) { return makeError(reset.error()); }
        auto cull = instanceCullKernel_.prepareDispatch(*encoded,
            std::max(divideRoundUp(static_cast<uint32_t>(streamRuntime_->asset().instances().size()), 64u), 1u));
        if (!cull) { return makeError(cull.error()); }
        const GPUSceneInstanceCullRecordDesc desc{.phase = phase, .reset = std::move(*reset), .cull = std::move(*cull)};
        std::string log;
        Result<> result = gpuSceneSubsystem_->recordInstanceCull(
            commandBuffer,
            gpuSceneView_,
            activeFrameSlot_,
            desc,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenStreamAssetPass] {}", log);
        }
        return result;
    }

    Result<> buildHzb(CommandBuffer& commandBuffer, TextureView* depth, bool reversedZ)
    {
        if (gpuSceneSubsystem_ == nullptr || hzbMipCount_ == 0) {
            return makeError(Error::InvalidArgument);
        }
        const auto* visible = gpuSceneSubsystem_->visibleDrawSet(gpuSceneView_, activeFrameSlot_);
        if (visible == nullptr) { return makeError(Error::InvalidArgument); }
        const auto& hzb = visible->gpu.hzb;
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        std::vector<PreparedComputeDispatch> dispatches;
        dispatches.reserve(hzbMipCount_);
        uint32_t mipWidth = frameWidth_, mipHeight = frameHeight_;
        uint32_t sourceWidth = mipWidth, sourceHeight = mipHeight;
        uint32_t sourceOffset = 0, destinationOffset = 0;
        for (uint32_t mipLevel = 0; mipLevel < hzbMipCount_; ++mipLevel) {
            ParameterWriter writer(*device_, **registry, commandBuffer.frameContext());
            const HZBParameters params{
                .hzb = writer.dataBuffer(hzb.history[hzb.writeIndex].buffer, sizeof(float), alignof(float)),
                .depth = writer.sampledImage(depth),
                .width = mipWidth, .height = mipHeight,
                .sourceWidth = sourceWidth, .sourceHeight = sourceHeight,
                .sourceOffset = sourceOffset, .destinationOffset = destinationOffset,
                .reversedZ = reversedZ ? 1u : 0u, .mipLevel = mipLevel,
            };
            auto encoded = writer.encode(params, kHZBABI, ParameterTransport::InlinePush);
            if (!encoded) { return makeError(encoded.error()); }
            auto dispatch = hzbKernel_.prepareDispatch(*encoded, divideRoundUp(mipWidth, 8u), divideRoundUp(mipHeight, 8u));
            if (!dispatch) { return makeError(dispatch.error()); }
            dispatches.push_back(std::move(*dispatch));
            sourceWidth = mipWidth; sourceHeight = mipHeight;
            sourceOffset = destinationOffset;
            destinationOffset += mipWidth * mipHeight;
            mipWidth = std::max(1u, (mipWidth + 1u) / 2u);
            mipHeight = std::max(1u, (mipHeight + 1u) / 2u);
        }
        const GPUSceneHZBRecordDesc desc{.preparedDispatches = dispatches};
        std::string log;
        Result<> result = gpuSceneSubsystem_->recordBuildHzb(
            commandBuffer,
            gpuSceneView_,
            activeFrameSlot_,
            desc,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenStreamAssetPass] {}", log);
        }
        return result;
    }

    Result<> initializeRayQuery(Device& device, std::string& log)
    {
        const char* capabilities[] = {
            "spvRayQueryKHR",
            "SPV_NV_cluster_acceleration_structure",
            "spvRayTracingClusterAccelerationStructureNV",
        };
        const SlangMacroDefine macros[] = {
            SlangMacroDefine{
                .name = "SCENE_RAYQUERY_ENABLE_CLUSTER_ID",
                .value = "1",
            },
        };
        ShaderCompileResult compileResult;
        Result<> result = compileSlangShaderToSpirv(SlangShaderDesc{
            .moduleName = kSceneRayQueryVisualizationShaderModuleName,
            .entryPointName = kSceneRayQueryVisualizationEntryPoint,
            .searchPath = kTriangleShaderSearchPath,
            .capabilities = {capabilities, static_cast<uint32_t>(std::size(capabilities))},
            .macroDefines = {macros, static_cast<uint32_t>(std::size(macros))},
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
        if (!result) {
            log += "compileSlangShaderToSpirv(stream RTAS visualization) returned ";
            log += resultToString(result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            log += '\n';
            return result;
        }

        return rayQueryProgram_.initialize(
            device,
            ComputeKernelDesc{
                .spirv = compileResult.spirv,
                .parameters = parameterAbi<SceneRayQueryVisualizationParams>(kSceneRayQueryVisualizationABI, ParameterTransport::InlinePush),
                .debugName = "GPUDrivenStreamAssetPass RTAS visualization",
            },
            log);
    }

    MeshletStreamFrameDesc frameDescFromContext(const RenderGraphExecutionContext& context) const
    {
        const scene::Bounds& bounds = streamRuntime_->bounds();
        const float3 center = bounds.center();
        const float radius = std::max(bounds.radius(), 1.0f);
        const RenderGraphProperties* camera = cameraPropertiesFrom(context.properties());
        const float3 defaultEye(center.x, center.y + radius * 0.35f, center.z + radius * 2.5f);
        const float3 eye = cameraVec3(camera, "eye", defaultEye);
        const float3 lookAt = cameraVec3(camera, "center", center);
        const float3 up = cameraVec3(camera, "up", float3(0.0f, 1.0f, 0.0f));
        const float fovDegrees = cameraFloat(camera, "fovDegrees", 60.0f);
        const float znear = cameraFloat(camera, "znear", 0.1f);
        const float zfar = cameraFloat(camera, "zfar", std::max(radius * 8.0f, znear + 100.0f));
        const bool enableGpuLodSelection = autoLodProperty(context.properties());
        const auto projection = camera != nullptr ? camera->find("projection") : context.properties().end();
        const bool orthographic = camera != nullptr && projection != camera->end() && projection->is_string() &&
            (projection->get<std::string>() == "orthographic" || projection->get<std::string>() == "ortho");

        MeshletStreamFrameDesc frame{
            .width = context.width(),
            .height = context.height(),
            .displayHeight = context.displayHeight(),
            .selectedLodLevel = enableGpuLodSelection
                ? kMeshletStreamNoDebugLODOverride
                : selectedLodProperty(context.properties()),
            .enableGpuLodSelection = enableGpuLodSelection,
            .lodPixelError = cameraFloat(&context.properties(), "lodPixelError", 1.5f),
            .lodBias = cameraFloat(&context.properties(), "lodBias", 0.0f),
            .debugColorMode = debugColorModeFromProperties(context.properties()),
            .camera = MeshletStreamCameraDesc{
                .eye = eye,
                .center = lookAt,
                .up = up,
                .fovDegrees = fovDegrees,
                .znear = znear,
                .zfar = zfar,
                .orthographic = orthographic,
                .reversedZ = camera != nullptr ? boolProperty(*camera, "reversedZ", kDefaultReversedZ) : kDefaultReversedZ,
                .orthoHeight = cameraFloat(camera, "orthoHeight", 10.0f),
            },
        };
        if (const ViewConstants* view = context.viewConstants()) {
            const auto& current = view->current;
            frame.camera.eye = float3(current.eye[0], current.eye[1], current.eye[2]);
            frame.camera.center = float3(current.center[0], current.center[1], current.center[2]);
            frame.camera.up = float3(current.upProjection[0], current.upProjection[1], current.upProjection[2]);
            frame.camera.fovDegrees = current.viewport[3] * (180.0f / 3.14159265359f);
            frame.camera.znear = current.clipOrtho[0];
            frame.camera.zfar = current.clipOrtho[1];
            frame.camera.orthographic = current.upProjection[3] > 0.5f;
            frame.camera.orthoHeight = current.clipOrtho[2];
            frame.camera.reversedZ = current.clipOrtho[3] > 0.5f;
            frame.jitterX = view->jitter[0];
            frame.jitterY = view->jitter[1];
        }
        return frame;
    }

    Result<> draw(
        RenderGraphExecutionContext& context,
        TextureView& visibility,
        TextureHandle depth,
        GPUSceneCullPhase phase,
        LoadOp loadOp,
        bool reversedZ)
    {
        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = &visibility,
            .state = ResourceState::ColorAttachment,
            .loadOp = loadOp,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 0.0f},
        };
        RenderingAttachmentDesc depthAttachment{
            .view = depth.view(),
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = loadOp,
            .storeOp = StoreOp::Store,
            .clearDepth = depthClearValue(reversedZ),
        };
        if (auto rendering = context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = {&attachment, 1},
            .depthStencilAttachment = &depthAttachment,
        }); !rendering) { return rendering; }
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        if (streamRuntime_->drawTaskCount() > 0) {
            context.commandBuffer().bindBindlessHeap(*streamRuntime_->bindlessHeap());
            if (auto commandResult = context.commandBuffer().bindExecution((visibilityPipelines_[reversedZ ? 1u : 0u])->execution()); !commandResult) { return commandResult; }
            auto registry = device_->resourceRegistry();
            if (!registry) { return makeError(registry.error()); }
            auto& commands = context.commandBuffer();
            ParameterWriter writer(*device_, **registry, commands.frameContext());
            auto push = streamRuntime_->hardwareParameters(writer);
            push.traversalPhase = phase == GPUSceneCullPhase::Early ? 0u : 1u;
            auto encoded = writer.encode(push, kStreamHardwareABI, ParameterTransport::InlinePush);
            if (!encoded) { return makeError(encoded.error()); }
            if (auto retained = encoded->bindResources(commands); !retained) { return retained; }
            commands.pushBindlessData(encoded->inlineData().data(), static_cast<uint32_t>(encoded->inlineData().size()));
            streamRuntime_->cmdDrawMeshTasks(context.commandBuffer());
        }
        context.commandBuffer().endRendering();
        return {};
    }

    Result<> dispatchDeferred(RenderGraphExecutionContext& context)
    {
        Result<> result = streamRuntime_->cmdPrepareDeferred(context.commandBuffer());
        if (!result) {
            return result;
        }
        const auto resources = streamRuntime_->deferredGpuResources();
        if (!resources.valid()) { return makeError(Error::InvalidArgument); }
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, commands.frameContext());
        const StreamDeferredParameters params{
            .settings = writer.dataBuffer(resources.paramsBuffer, sizeof(MeshletStreamGPUParams), 16),
            .records = writer.dataBuffer(resources.visibleClusterBuffer, sizeof(CompactStreamVisibleRecord), 4),
            .groups = writer.dataBuffer(resources.activeGroupBuffer, sizeof(MeshletStreamGPUActiveGroup), 16),
            .pageTable = writer.dataBuffer(resources.pageTableBuffer, sizeof(StreamPageTableEntry), 8),
            .header = writer.dataBuffer(resources.activeHeaderBuffer, sizeof(MeshletStreamGPUActiveHeader), 4),
            .output = writer.dataBuffer(deferredColorBuffer_.get(), 4, 4),
            .pages = writer.dataBuffer(resources.pageBuffer, 4, 4),
            .visibility = writer.sampledImage(context.outputTexture("visibility").view()),
            .width = frameWidth_, .height = frameHeight_,
            .recordBase = 0, .recordCapacity = resources.visibleRecordCapacity,
        };
        auto encoded = writer.encode(params, kStreamDeferredABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return deferredKernel_.dispatch(commands, *encoded, (frameWidth_ + 7u) / 8u, (frameHeight_ + 7u) / 8u);
    }

    Result<> drawComposite(RenderGraphExecutionContext& context, TextureHandle color)
    {
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, commands.frameContext());
        const StreamCompositeParameters params{
            .colors = writer.dataBuffer(deferredColorBuffer_.get(), 4, 4),
            .width = frameWidth_, .height = frameHeight_,
        };
        auto encoded = writer.encode(params, kStreamCompositeABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        if (auto result = encoded->bindResources(commands); !result) { return result; }
        const auto bytes = encoded->inlineData();
        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = frameWidth_,
            .height = frameHeight_,
        };
        RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.015f, 0.018f, 0.024f, 1.0f},
        };
        if (auto rendering = context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = {&attachment, 1},
        }); !rendering) { return rendering; }
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(frameWidth_),
            .height = static_cast<float>(frameHeight_),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        if (auto result = commands.bindExecution(compositePipeline_->execution(), bytes.data(), uint32_t(bytes.size())); !result) {
            commands.endRendering();
            return result;
        }
        context.commandBuffer().draw(3u, 1u, 0u, 0u);
        context.commandBuffer().endRendering();
        return {};
    }

    Result<> drawRayQuery(
        RenderGraphExecutionContext& context,
        TextureHandle color,
        const MeshletStreamFrameDesc& frame)
    {
        if (!streamRuntime_->tlasReady() ||
            streamRuntime_->accelerationStructure() == nullptr ||
            color.view() == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        SceneRayQueryVisualizationPush push{};
        push.eye[0] = frame.camera.eye.x;
        push.eye[1] = frame.camera.eye.y;
        push.eye[2] = frame.camera.eye.z;
        push.eye[3] = 0.0f;
        push.center[0] = frame.camera.center.x;
        push.center[1] = frame.camera.center.y;
        push.center[2] = frame.camera.center.z;
        push.center[3] = 0.0f;
        push.upProjection[0] = frame.camera.up.x;
        push.upProjection[1] = frame.camera.up.y;
        push.upProjection[2] = frame.camera.up.z;
        push.upProjection[3] = frame.camera.orthographic ? 1.0f : 0.0f;
        push.viewport[0] = static_cast<float>(std::max(context.width(), 1u)) /
            static_cast<float>(std::max(context.height(), 1u));
        push.viewport[1] = static_cast<float>(context.width());
        push.viewport[2] = static_cast<float>(context.height());
        push.viewport[3] = frame.camera.fovDegrees * 0.017453292519943295f;
        push.clipOrtho[0] = frame.camera.znear;
        push.clipOrtho[1] = frame.camera.zfar;
        push.clipOrtho[2] = std::max(frame.camera.orthoHeight, 0.0001f);
        push.clipOrtho[3] = 0.0f;
        push.mode = rtasGranularityFromProperties(context.properties());
        push.width = context.width();
        push.height = context.height();

        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, commands.frameContext());
        const SceneRayQueryVisualizationParams params{
            .scene = writer.accelerationStructure(streamRuntime_->accelerationStructure()),
            .output = writer.storageImage(color.view()),
            .settings = push,
        };
        auto encoded = writer.encode(params, kSceneRayQueryVisualizationABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return rayQueryProgram_.dispatch(commands, *encoded, (context.width() + 7u) / 8u, (context.height() + 7u) / 8u);
    }

    std::unique_ptr<PipelineCache> pipelineCache_;
    std::shared_ptr<MeshletStreamRuntime> streamRuntime_;
    std::unique_ptr<ShaderModule> meshShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<ShaderModule> compositeVertexShader_;
    std::unique_ptr<ShaderModule> compositeFragmentShader_;
    std::array<std::unique_ptr<GraphicsPipeline>, 2> visibilityPipelines_;
    ComputeKernel deferredKernel_;
    std::unique_ptr<GraphicsPipeline> compositePipeline_;
    ComputeKernel cullResetKernel_;
    ComputeKernel instanceCullKernel_;
    ComputeKernel hzbKernel_;
    std::unique_ptr<Buffer> deferredColorBuffer_;
    ResourceLease visibilityImageHandle_;
    ResourceLease depthImageHandle_;
    ResourceLease instanceVisibilityHandle_;
    ResourceLease visibleInstanceIdsHandle_;
    ResourceLease visibleInstanceCounterHandle_;
    std::array<ResourceLease, 2> hzbHandles_{};
    uint32_t frameWidth_ = 1;
    uint32_t frameHeight_ = 1;
    uint32_t hzbMipCount_ = 1;
    uint64_t hzbElementCount_ = 1;
    uint32_t frameSlotCount_ = 0;
    uint32_t activeFrameSlot_ = 0;
    uint32_t instanceCapacity_ = 1;
    uint64_t observedHistoryInvalidationRevision_ = 0;
    Device* device_ = nullptr;
    GPUSceneSubsystem* gpuSceneSubsystem_ = nullptr;
    GPUSceneViewId gpuSceneView_;
    ClusterLightGridDesc lightGridDesc_;
    const scene::Scene* gpuSceneSource_ = nullptr;
    GPUSceneSourceOverrideToken gpuSceneSourceToken_;
    bool hzbValid_ = false;
    ComputeKernel rayQueryProgram_;
    bool rtasVisualization_ = false;
    uint64_t compiledSourceIdentity_ = 0;
    uint64_t compiledSourceContentRevision_ = 0;
    Format compiledColorFormat_ = Format::RGBA8Unorm;
    bool compiledDebugReadback_ = false;
    bool compiledStreamAssetOnly_ = false;
    bool compiled_ = false;
};

std::unique_ptr<RenderGraphPass> createGPUDrivenStreamAssetPass()
{
    return std::make_unique<GPUDrivenStreamAssetPass>();
}

} // namespace metallic::render::builtin_pass
