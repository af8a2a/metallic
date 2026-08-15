#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSceneRtx.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetConfig.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
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

std::filesystem::path streamAssetPathFromProperties(
    const RenderGraphProperties& props,
    const std::filesystem::path& scenePath)
{
    return pathFromProperties(
        props,
        "streamAssetPath",
        scene::meshletStreamAssetPathFor(scenePath));
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

uint64_t uint64Property(const RenderGraphProperties& props, const char* key, uint64_t fallback)
{
    auto iter = props.find(key);
    if (iter == props.end() || !iter->is_number_integer()) {
        return fallback;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 ? fallback : static_cast<uint64_t>(value);
}

uint32_t selectedLodProperty(const RenderGraphProperties& props)
{
    auto iter = props.find("selectedLodLevel");
    if (iter == props.end() || !iter->is_number_integer()) {
        return 0;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 ? 0u : static_cast<uint32_t>(std::min<int64_t>(value, std::numeric_limits<uint32_t>::max()));
}

uint32_t debugColorModeFromProperties(const RenderGraphProperties& props)
{
    auto iter = props.find("debugColorMode");
    if (iter == props.end() || !iter->is_string()) {
        return kMeshletStreamDebugShaded;
    }
    const std::string mode = iter->get<std::string>();
    if (mode == "lod") {
        return kMeshletStreamDebugLod;
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

MeshletStreamRuntimeDesc runtimeDescFromProperties(const RenderGraphProperties& properties)
{
    const std::filesystem::path scenePath = scenePathFromProperties(properties);
    const uint32_t maxGpuPageRequests = std::max<uint32_t>(
        uintProperty(properties, "maxGpuPageRequests", kMeshletStreamDefaultMaxGpuPageRequests),
        1u);
    return MeshletStreamRuntimeDesc{
        .sourcePath = scenePath,
        .streamAssetPath = streamAssetPathFromProperties(properties, scenePath),
        .autoBuildStreamAsset = boolProperty(properties, "autoBuildStreamAsset", false),
        .maxResidentBytes = uint64Property(properties, "maxResidentBytes", 0),
        .maxResidentPages = uintProperty(properties, "maxResidentPages", 4096),
        .maxLockedFallbackPages = uintProperty(properties, "maxLockedFallbackPages", 1024),
        .maxPageUploadsPerFrame = uintProperty(properties, "maxPageUploadsPerFrame", 64),
        .maxGpuPageRequests = maxGpuPageRequests,
        .maxGpuPageUnloadRequests = std::max<uint32_t>(
            uintProperty(properties, "maxGpuPageUnloadRequests", maxGpuPageRequests),
            1u),
        .maxActiveGroups = std::max<uint32_t>(
            uintProperty(properties, "maxActiveGroups", kMeshletStreamDefaultMaxActiveGroups),
            1u),
        .maxTraversalWorkers = std::max<uint32_t>(
            uintProperty(properties, "maxTraversalWorkers", kMeshletStreamDefaultTraversalWorkers),
            1u),
        .maxTraversalWorkItems = std::min(
            std::max<uint32_t>(
                uintProperty(properties, "maxTraversalWorkItems", kMeshletStreamDefaultTraversalWorkItems),
                1u),
            kMeshletStreamMaxTraversalWorkItems),
        .pageLoadConcurrency = pageLoadConcurrencyFromProperties(properties),
        .maxPageLoadsInFlight = std::max<uint32_t>(
            uintProperty(properties, "maxPageLoadsInFlight", 128),
            1u),
        .queuedFrameCount = 3,
        .enableClusterRtx = boolProperty(properties, "enableClusterRtx", false),
        .maxClasBytes = uint64Property(properties, "maxClasBytes", 512ull * 1024ull * 1024ull),
        .maxClasBuildClusters = uintProperty(properties, "maxClasBuildClusters", 0),
        .maxBlasClusterReferences = uintProperty(properties, "maxBlasClusterReferences", 0),
        .maxBlasBytes = uint64Property(properties, "maxBlasBytes", 512ull * 1024ull * 1024ull),
        .maxBlasBuilds = std::max<uint32_t>(
            uintProperty(properties, "maxBlasBuilds", kMeshletStreamDefaultMaxBlasBuilds),
            1u),
        .maxFallbackBlasBytes = uint64Property(
            properties,
            "maxFallbackBlasBytes",
            512ull * 1024ull * 1024ull),
    };
}

Result createMeshShader(Device& device, std::unique_ptr<ShaderModule>& outShader, std::string& log)
{
    ShaderCompileResult meshCompile;
    const char* capabilities[] = {"spvMeshShadingEXT"};
    Result result = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamMeshEntryPoint,
            .searchPath = kMeshletStreamShaderSearchPath,
            .capabilities = capabilities,
            .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
        },
        meshCompile);
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

    result = device.createShaderModule(
        ShaderModuleDesc{
            .code = meshCompile.spirv.data(),
            .byteSize = static_cast<uint64_t>(meshCompile.spirv.size() * sizeof(uint32_t)),
        },
        outShader);
    if (!result || outShader == nullptr) {
        log += resultMessage("createShaderModule(GPUDrivenStreamAsset mesh)", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

Result createStreamShader(
    Device& device,
    const char* entryPoint,
    std::unique_ptr<ShaderModule>& outShader,
    std::string& log)
{
    ShaderCompileResult compileResult;
    Result result = compileSlangShader(
        kMeshletStreamShaderModuleName,
        entryPoint,
        compileResult,
        log);
    if (!result) {
        return result;
    }
    result = device.createShaderModule(
        ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
        },
        outShader);
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
    ~GPUDrivenStreamAssetPass() override
    {
        releaseGPUSceneSourceLease();
        if (gpuSceneSubsystem_ != nullptr && gpuSceneView_.valid()) {
            gpuSceneSubsystem_->destroyView(gpuSceneView_);
        }
    }

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{
            GPUSceneSubsystem::kSubsystemId,
        };
        return required;
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        RenderGraphField& color = reflection.addTextureOutput(
            "color",
            "Meshlet streamasset deferred color");
        if (boolProperty(properties(), "rtasVisualization", false)) {
            color.texture2D().storageReadWrite().format = Format::Rgba8Unorm;
        } else {
            color.texture2D().colorWrite();
        }
        RenderGraphField& visibility = reflection.addTextureOutput(
            "visibility",
            "Stream mesh shader visibility IDs");
        visibility.colorWrite();
        visibility.format = Format::R32Uint;
        visibility.usage = visibility.usage | TextureUsageBits::Sampled;
        RenderGraphField& depth = reflection.addTextureOutput(
            "depth",
            "Meshlet streamasset visibility depth and HZB source");
        depth.texture2D().depthStencilWrite();
        depth.usage = depth.usage | TextureUsageBits::Sampled;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeBoolSetting("enableGpuLodSelection", "GPU LOD", true),
            runtimeBoolSetting("instanceFrustumCull", "Instance Frustum Cull", true),
            runtimeBoolSetting("instanceHzbCull", "Instance HZB Cull", true),
            runtimeBoolSetting("clusterFrustumCull", "Cluster Sphere / Frustum Cull", true),
            runtimeBoolSetting("clusterNormalConeCull", "Cluster Normal Cone Cull", true),
            runtimeIntSetting("selectedLodLevel", "LOD", 0, 0, 31),
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

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        log.clear();
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().meshShader ||
            !context.device->capabilities().bindlessDescriptorHeap) {
            log = "GPUDrivenStreamAssetPass requires meshShader and bindlessDescriptorHeap capabilities";
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

        const scene::Scene* runtimeScene = runtimeSceneForPath(
            context.runtimeScene,
            scenePathFromProperties(properties()));
        if (runtimeScene == nullptr && context.sceneResourceManager != nullptr) {
            Result sceneResult = context.sceneResourceManager->resolveScene(
                properties(),
                context.runtimeScene,
                runtimeScene,
                log);
            if (!sceneResult) {
                return sceneResult;
            }
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

        Result result = streamRuntime_.initialize(
            *context.device,
            runtimeDescFromProperties(properties()),
            log);
        if (!result) {
            return result;
        }
        rtasVisualization_ = boolProperty(properties(), "rtasVisualization", false);
        if (rtasVisualization_) {
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
            kMeshletStreamDeferredEntryPoint,
            deferredShader_,
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
        result = createStreamShader(
            *context.device,
            kMeshletStreamCullResetEntryPoint,
            cullResetShader_,
            log);
        if (!result) {
            return result;
        }
        result = createStreamShader(
            *context.device,
            kMeshletStreamInstanceCullEntryPoint,
            instanceCullShader_,
            log);
        if (!result) {
            return result;
        }
        result = createStreamShader(
            *context.device,
            kMeshletStreamHzbEntryPoint,
            hzbShader_,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .meshShader = meshShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = Format::R32Uint,
                .depthStencilFormat = Format::D32Sfloat,
                .depthStencil = DepthStencilState{
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = depthCompareOp(kDefaultReversedZ),
                },
                .usesBindlessHeap = true,
            },
            visibilityPipeline_);
        if (!result || visibilityPipeline_ == nullptr) {
            log += resultMessage("createGraphicsPipeline(GPUDrivenStreamAsset visibility)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = context.device->createComputePipeline(
            ComputePipelineDesc{
                .computeShader = deferredShader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            },
            deferredPipeline_);
        if (!result || deferredPipeline_ == nullptr) {
            log += resultMessage("createComputePipeline(GPUDrivenStreamAsset deferred)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .vertexShader = compositeVertexShader_.get(),
                .fragmentShader = compositeFragmentShader_.get(),
                .colorFormat = context.defaultFormat,
                .usesBindlessHeap = true,
            },
            compositePipeline_);
        if (!result || compositePipeline_ == nullptr) {
            log += resultMessage("createGraphicsPipeline(GPUDrivenStreamAsset composite)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        auto createComputePipeline = [&](ShaderModule& shader,
                                         std::unique_ptr<ComputePipeline>& pipeline,
                                         const char* label) -> Result {
            Result pipelineResult = context.device->createComputePipeline(
                ComputePipelineDesc{
                    .computeShader = &shader,
                    .computeEntryPoint = "main",
                    .usesBindlessHeap = true,
                    .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
                },
                pipeline);
            if (!pipelineResult || pipeline == nullptr) {
                log += resultMessage(
                    std::string("createComputePipeline(GPUDrivenStreamAsset ") + label + ")",
                    pipelineResult);
                log += '\n';
                return pipelineResult ? makeError(Error::Failure) : pipelineResult;
            }
            return {};
        };
        result = createComputePipeline(*cullResetShader_, cullResetPipeline_, "cull reset");
        if (!result) {
            return result;
        }
        result = createComputePipeline(*instanceCullShader_, instanceCullPipeline_, "instance cull");
        if (!result) {
            return result;
        }
        result = createComputePipeline(*hzbShader_, hzbPipeline_, "HZB");
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
            gpuSceneView_ = gpuSceneSubsystem_->createView(GPUSceneViewDesc{
                .frameSlotCount = requestedFrameSlotCount,
            });
            if (!gpuSceneView_.valid()) {
                log = "GPUDrivenStreamAssetPass failed to allocate a GPUScene View";
                return makeError(Error::Failure);
            }
        }
        frameSlotCount_ = requestedFrameSlotCount;
        instanceCapacity_ = std::max<uint32_t>(
            static_cast<uint32_t>(streamRuntime_.asset().instances().size()),
            1u);
        for (const scene::MeshletStreamInstanceInfo& instance :
             streamRuntime_.asset().instances()) {
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

        result = context.device->createBuffer(
            BufferDesc{
                .size = static_cast<uint64_t>(frameWidth_) * frameHeight_ * sizeof(uint32_t),
                .structureStride = sizeof(uint32_t),
                .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::Device,
            },
            deferredColorBuffer_);
        if (!result || deferredColorBuffer_ == nullptr) {
            log += resultMessage("createBuffer(GPUDrivenStreamAsset deferred color)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        BindlessHeap* heap = streamRuntime_.bindlessHeap();
        result = heap->allocateSampledImage(visibilityImageHandle_);
        if (!result || !visibilityImageHandle_.valid()) {
            log += resultMessage("allocateSampledImage(GPUDrivenStreamAsset visibility)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = heap->allocateSampledImage(depthImageHandle_);
        if (!result || !depthImageHandle_.valid()) {
            log += resultMessage("allocateSampledImage(GPUDrivenStreamAsset depth)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = heap->allocateBuffer(deferredColorHandle_);
        if (!result || !deferredColorHandle_.valid()) {
            log += resultMessage("allocateBuffer(GPUDrivenStreamAsset deferred color)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = heap->writeStorageBuffer(deferredColorHandle_, *deferredColorBuffer_);
        if (!result) {
            log += resultMessage("writeStorageBuffer(GPUDrivenStreamAsset deferred color)", result);
            log += '\n';
            return result;
        }

        auto allocateGpuSceneBufferHandle = [&](BindlessHandle& handle,
                                                const char* label) -> Result {
            Result allocateResult = heap->allocateBuffer(handle);
            if (!allocateResult || !handle.valid()) {
                log += resultMessage(
                    std::string("allocateBuffer(GPUDrivenStreamAsset ") + label + ")",
                    allocateResult);
                log += '\n';
                return allocateResult ? makeError(Error::Failure) : allocateResult;
            }
            return {};
        };
        result = allocateGpuSceneBufferHandle(
            instanceVisibilityHandle_,
            "instance visibility");
        if (!result) {
            return result;
        }
        result = allocateGpuSceneBufferHandle(
            visibleInstanceIdsHandle_,
            "visible instance IDs");
        if (!result) {
            return result;
        }
        result = allocateGpuSceneBufferHandle(
            visibleInstanceCounterHandle_,
            "visible instance counter");
        if (!result) {
            return result;
        }
        for (uint32_t historyIndex = 0; historyIndex < hzbHandles_.size(); ++historyIndex) {
            result = allocateGpuSceneBufferHandle(
                hzbHandles_[historyIndex],
                "HZB history");
            if (!result) {
                return result;
            }
        }
        result = streamRuntime_.updateRasterBindings(MeshletStreamGpuRasterBindings{
            .instanceVisibilityBuffer = instanceVisibilityHandle_.index,
            .hzbBuffer0 = hzbHandles_[0].index,
            .hzbBuffer1 = hzbHandles_[1].index,
            .depthImage = depthImageHandle_.index,
            .visibilityImage = visibilityImageHandle_.index,
            .deferredColorBuffer = deferredColorHandle_.index,
            .visibleInstanceIdsBuffer = visibleInstanceIdsHandle_.index,
            .hzbMipCount = hzbMipCount_,
            .hzbValid = 0u,
            .cullingFlags = cullingFlagsFromProperties(properties()),
            .width = frameWidth_,
            .height = frameHeight_,
            .visibleInstanceCounterBuffer = visibleInstanceCounterHandle_.index,
        });
        if (!result) {
            log = "GPUDrivenStreamAssetPass failed to publish raster bindings";
            return result;
        }
        deferredColorState_ = ResourceState::Undefined;

        if (rtasVisualization_) {
            result = initializeRayQuery(*context.device, log);
            if (!result) {
                return result;
            }
        }

        device_ = context.device;
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        GPUSceneSubsystem* gpuSceneSubsystem = context.subsystem<GPUSceneSubsystem>();
        if (gpuSceneSubsystem == nullptr ||
            gpuSceneSubsystem != gpuSceneSubsystem_ ||
            !gpuSceneView_.valid()) {
            return makeError(Error::InvalidArgument);
        }
        const scene::Scene* runtimeScene = runtimeSceneForPath(
            context.runtimeScene(),
            scenePathFromProperties(context.properties()));
        if (runtimeScene != nullptr) {
            std::string syncLog;
            Result syncResult = streamRuntime_.syncRuntimeScene(*runtimeScene, syncLog);
            if (!syncResult) {
                spdlog::warn("[GPUDrivenStreamAssetPass] Runtime scene sync failed: {}", syncLog);
                return syncResult;
            }
        }

        std::vector<uint32_t> gpuSceneInstanceMapping(
            streamRuntime_.asset().instances().size(),
            std::numeric_limits<uint32_t>::max());
        uint32_t mappedInstanceCount = 0;
        for (size_t instanceIndex = 0;
             instanceIndex < streamRuntime_.asset().instances().size();
             ++instanceIndex) {
            const GPUSceneInstanceId gpuSceneInstance =
                gpuSceneSubsystem->instanceForRenderNode(
                    streamRuntime_.asset().instances()[instanceIndex].renderNodeIndex);
            if (gpuSceneInstance.valid()) {
                gpuSceneInstanceMapping[instanceIndex] = gpuSceneInstance.index;
                ++mappedInstanceCount;
            }
        }
        if (!gpuSceneInstanceMapping.empty() && mappedInstanceCount == 0) {
            spdlog::warn(
                "[GPUDrivenStreamAssetPass] GPUScene mapping is empty for {} stream instances (GPUScene instances={})",
                gpuSceneInstanceMapping.size(),
                gpuSceneSubsystem->instances().size());
        }
        Result result = streamRuntime_.syncGPUSceneInstanceMapping(
            gpuSceneInstanceMapping);
        if (!result) {
            return result;
        }

        TextureHandle color = context.outputTexture("color");
        TextureHandle visibility = context.outputTexture("visibility");
        TextureHandle depth = context.outputTexture("depth");
        if (!color.valid() ||
            !visibility.valid() ||
            !depth.valid() ||
            context.streamer() == nullptr ||
            !streamRuntime_.ready() ||
            streamRuntime_.bindlessHeap() == nullptr ||
            visibilityPipeline_ == nullptr ||
            deferredPipeline_ == nullptr ||
            compositePipeline_ == nullptr ||
            cullResetPipeline_ == nullptr ||
            instanceCullPipeline_ == nullptr ||
            hzbPipeline_ == nullptr ||
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
        result = streamRuntime_.bindlessHeap()->writeSampledImage(
            visibilityImageHandle_,
            *visibility.view(),
            ResourceState::ShaderRead);
        if (!result) {
            return result;
        }
        result = streamRuntime_.bindlessHeap()->writeSampledImage(
            depthImageHandle_,
            *depth.view(),
            ResourceState::ShaderRead);
        if (!result) {
            return result;
        }
        result = streamRuntime_.cmdBeginFrame(context.commandBuffer(), *context.streamer(), frame);
        if (!result) {
            return result;
        }
        if (rtasVisualization_) {
            result = streamRuntime_.cmdPreTraversal(context.commandBuffer(), frame);
            if (!result) {
                return result;
            }
            result = drawRayQuery(context, color, frame);
        } else {
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

            // Stream traversal currently also publishes the per-frame camera and
            // scene params consumed by the cull kernels. P2 can split that upload
            // from traversal so the early instance state also prunes page demand.
            result = streamRuntime_.cmdPreTraversal(context.commandBuffer(), frame);
            if (!result) {
                return result;
            }
            result = dispatchInstanceCull(
                context.commandBuffer(),
                GPUSceneCullPhase::Early);
            if (!result) {
                return result;
            }
            result = streamRuntime_.cmdPrepareVisibility(context.commandBuffer());
            if (result) {
                result = draw(
                    context,
                    *visibility.view(),
                    depth,
                    GPUSceneCullPhase::Early,
                    LoadOp::Clear);
            }
            if (result) {
                transitionTexture(
                    context.commandBuffer(),
                    *depth.texture(),
                    ResourceState::DepthStencilAttachment,
                    ResourceState::ShaderRead);
                result = buildHzb(context.commandBuffer());
            }
            if (result) {
                transitionTexture(
                    context.commandBuffer(),
                    *depth.texture(),
                    ResourceState::ShaderRead,
                    ResourceState::DepthStencilAttachment);
                result = dispatchInstanceCull(
                    context.commandBuffer(),
                    GPUSceneCullPhase::Late);
            }
            if (result) {
                result = streamRuntime_.cmdPrepareVisibility(context.commandBuffer());
            }
            if (result) {
                result = draw(
                    context,
                    *visibility.view(),
                    depth,
                    GPUSceneCullPhase::Late,
                    LoadOp::Load);
            }
            if (result) {
                transitionTexture(
                    context.commandBuffer(),
                    *depth.texture(),
                    ResourceState::DepthStencilAttachment,
                    ResourceState::ShaderRead);
                result = buildHzb(context.commandBuffer());
            }
            if (result) {
                result = dispatchDeferred(context, visibility);
            }
            if (result) {
                transitionTexture(
                    context.commandBuffer(),
                    *depth.texture(),
                    ResourceState::ShaderRead,
                    ResourceState::DepthStencilAttachment);
                result = drawComposite(context, color);
            }
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
        result = streamRuntime_.cmdPostTraversal(context.commandBuffer());
        if (!result) {
            return result;
        }
        return streamRuntime_.cmdEndFrame(context.commandBuffer());
    }

private:
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

    Result ensureFrameResources(
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
            streamRuntime_.bindlessHeap() == nullptr ||
            !deferredColorHandle_.valid()) {
            return makeError(Error::InvalidArgument);
        }

        const uint32_t mipCount = computeHzbMipCount(width, height);
        const uint64_t elementCount = computeHzbElementCount(width, height, mipCount);
        std::unique_ptr<Buffer> resizedDeferredColorBuffer;
        Result result = device_->createBuffer(
            BufferDesc{
                .size = static_cast<uint64_t>(width) * height * sizeof(uint32_t),
                .structureStride = sizeof(uint32_t),
                .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::Device,
            },
            resizedDeferredColorBuffer);
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

        result = streamRuntime_.bindlessHeap()->writeStorageBuffer(
            deferredColorHandle_,
            *resizedDeferredColorBuffer);
        if (!result) {
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
        deferredColorState_ = ResourceState::Undefined;
        hzbValid_ = false;
        return {};
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

    Result prepareGPUSceneView(GPUSceneSubsystem& subsystem, bool cameraCut)
    {
        activeFrameSlot_ = subsystem.currentFrameSlot();
        if (activeFrameSlot_ >= frameSlotCount_) {
            return makeError(Error::InvalidArgument);
        }
        const GPUSceneViewPrepareInfo prepareInfo{
            .width = frameWidth_,
            .height = frameHeight_,
            .cameraCut = cameraCut,
            .freezeCullingCamera = false,
        };
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
        Result result = subsystem.publishViewGpuResources(
            gpuSceneView_,
            activeFrameSlot_,
            streamRuntime_.frameIndex() & 1u,
            log);
        if (!result) {
            spdlog::error("[GPUDrivenStreamAssetPass] {}", log);
        }
        return result;
    }

    Result bindGPUSceneViewResources(
        GPUSceneSubsystem& subsystem,
        TextureHandle depth)
    {
        GPUSceneViewGpuResourcesView resources;
        if (!depth.valid() ||
            !subsystem.viewGpuResources(
                gpuSceneView_,
                activeFrameSlot_,
                resources) ||
            resources.instanceVisibilityStates.view == nullptr ||
            resources.visibleInstanceIds.view == nullptr ||
            resources.visibleInstanceCounter.view == nullptr ||
            resources.hzbHistory[0].view == nullptr ||
            resources.hzbHistory[1].view == nullptr ||
            streamRuntime_.bindlessHeap() == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        BindlessHeap& heap = *streamRuntime_.bindlessHeap();
        Result result = heap.writeBufferView(
            instanceVisibilityHandle_,
            *resources.instanceVisibilityStates.view);
        if (result) {
            result = heap.writeBufferView(
                visibleInstanceIdsHandle_,
                *resources.visibleInstanceIds.view);
        }
        if (result) {
            result = heap.writeBufferView(
                visibleInstanceCounterHandle_,
                *resources.visibleInstanceCounter.view);
        }
        for (uint32_t historyIndex = 0;
             historyIndex < hzbHandles_.size() && result;
             ++historyIndex) {
            result = heap.writeBufferView(
                hzbHandles_[historyIndex],
                *resources.hzbHistory[historyIndex].view);
        }
        if (!result) {
            return result;
        }

        return streamRuntime_.updateRasterBindings(MeshletStreamGpuRasterBindings{
            .instanceVisibilityBuffer = instanceVisibilityHandle_.index,
            .hzbBuffer0 = hzbHandles_[0].index,
            .hzbBuffer1 = hzbHandles_[1].index,
            .depthImage = depthImageHandle_.index,
            .visibilityImage = visibilityImageHandle_.index,
            .deferredColorBuffer = deferredColorHandle_.index,
            .visibleInstanceIdsBuffer = visibleInstanceIdsHandle_.index,
            .hzbMipCount = hzbMipCount_,
            .hzbValid = hzbValid_ ? 1u : 0u,
            .cullingFlags = cullingFlagsFromProperties(properties()),
            .width = frameWidth_,
            .height = frameHeight_,
            .visibleInstanceCounterBuffer = visibleInstanceCounterHandle_.index,
        });
    }

    Result dispatchInstanceCull(
        CommandBuffer& commandBuffer,
        GPUSceneCullPhase phase)
    {
        if (gpuSceneSubsystem_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        MeshletStreamUserPush push = streamRuntime_.userPush();
        push.traversalPhase = phase == GPUSceneCullPhase::Early ? 0u : 1u;
        const GPUSceneInstanceCullRecordDesc desc{
            .phase = phase,
            .bindlessHeap = streamRuntime_.bindlessHeap(),
            .resetPipeline = cullResetPipeline_.get(),
            .instanceCullPipeline = instanceCullPipeline_.get(),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .instanceGroupCountX = std::max(
                divideRoundUp(
                    static_cast<uint32_t>(streamRuntime_.asset().instances().size()),
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
            spdlog::error("[GPUDrivenStreamAssetPass] {}", log);
        }
        return result;
    }

    Result buildHzb(CommandBuffer& commandBuffer)
    {
        if (gpuSceneSubsystem_ == nullptr || hzbMipCount_ == 0) {
            return makeError(Error::InvalidArgument);
        }
        std::vector<MeshletStreamUserPush> pushes;
        pushes.reserve(hzbMipCount_);
        for (uint32_t mipLevel = 0; mipLevel < hzbMipCount_; ++mipLevel) {
            MeshletStreamUserPush push = streamRuntime_.userPush();
            push.activeBuildPhase = mipLevel;
            pushes.push_back(push);
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
                .groupCountZ = 1u,
            });
            mipWidth = std::max(1u, (mipWidth + 1u) / 2u);
            mipHeight = std::max(1u, (mipHeight + 1u) / 2u);
        }
        const GPUSceneHzbRecordDesc desc{
            .bindlessHeap = streamRuntime_.bindlessHeap(),
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
            spdlog::error("[GPUDrivenStreamAssetPass] {}", log);
        }
        return result;
    }

    Result initializeRayQuery(Device& device, std::string& log)
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
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kSceneRayQueryVisualizationShaderModuleName,
                .entryPointName = kSceneRayQueryVisualizationEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
                .macroDefines = macros,
                .macroDefineCount = static_cast<uint32_t>(std::size(macros)),
            },
            compileResult);
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

        const vulkan::SceneRayQueryBindingDesc bindings[] = {
            vulkan::SceneRayQueryBindingDesc{
                .binding = 0,
                .kind = vulkan::SceneRayQueryBindingKind::AccelerationStructure,
            },
            vulkan::SceneRayQueryBindingDesc{
                .binding = 1,
                .kind = vulkan::SceneRayQueryBindingKind::StorageImage,
            },
        };
        return rayQueryProgram_.initialize(
            device,
            vulkan::SceneRayQueryProgramDesc{
                .spirv = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(SceneRayQueryVisualizationPush),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .debugName = "GPUDrivenStreamAssetPass RTAS visualization",
            },
            log);
    }

    MeshletStreamFrameDesc frameDescFromContext(const RenderGraphExecutionContext& context) const
    {
        const scene::Bounds& bounds = streamRuntime_.bounds();
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
        const bool enableGpuLodSelection = boolProperty(context.properties(), "enableGpuLodSelection", true);

        return MeshletStreamFrameDesc{
            .width = context.width(),
            .height = context.height(),
            .selectedLodLevel = enableGpuLodSelection
                ? kMeshletStreamNoDebugLodOverride
                : selectedLodProperty(context.properties()),
            .enableGpuLodSelection = enableGpuLodSelection,
            .debugColorMode = debugColorModeFromProperties(context.properties()),
            .camera = MeshletStreamCameraDesc{
                .eye = eye,
                .center = lookAt,
                .up = up,
                .fovDegrees = fovDegrees,
                .znear = znear,
                .zfar = zfar,
            },
        };
    }

    Result draw(
        RenderGraphExecutionContext& context,
        TextureView& visibility,
        TextureHandle depth,
        GPUSceneCullPhase phase,
        LoadOp loadOp)
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
            .clearDepth = depthClearValue(kDefaultReversedZ),
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = &depthAttachment,
        });
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        if (streamRuntime_.drawTaskCount() > 0) {
            context.commandBuffer().bindBindlessHeap(*streamRuntime_.bindlessHeap());
            context.commandBuffer().bindGraphicsPipeline(*visibilityPipeline_);
            MeshletStreamUserPush push = streamRuntime_.userPush();
            push.traversalPhase = phase == GPUSceneCullPhase::Early ? 0u : 1u;
            context.commandBuffer().pushBindlessData(&push, sizeof(push));
            streamRuntime_.cmdDrawMeshTasks(context.commandBuffer());
        }
        context.commandBuffer().endRendering();
        return {};
    }

    Result dispatchDeferred(
        RenderGraphExecutionContext& context,
        TextureHandle visibility)
    {
        TextureBarrierDesc visibilityBarrier{
            .texture = visibility.texture(),
            .before = ResourceState::ColorAttachment,
            .after = ResourceState::ShaderRead,
        };
        context.commandBuffer().barrier(BarrierDesc{
            .textures = &visibilityBarrier,
            .textureCount = 1,
        });
        BufferBarrierDesc colorBarrier{
            .buffer = deferredColorBuffer_.get(),
            .before = deferredColorState_,
            .after = ResourceState::General,
        };
        context.commandBuffer().barrier(BarrierDesc{
            .buffers = &colorBarrier,
            .bufferCount = 1,
        });
        deferredColorState_ = ResourceState::General;
        Result result = streamRuntime_.cmdPrepareDeferred(context.commandBuffer());
        if (!result) {
            return result;
        }
        context.commandBuffer().bindBindlessHeap(*streamRuntime_.bindlessHeap());
        context.commandBuffer().bindComputePipeline(*deferredPipeline_);
        const MeshletStreamUserPush push = streamRuntime_.userPush();
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
        context.commandBuffer().dispatch(
            (frameWidth_ + 7u) / 8u,
            (frameHeight_ + 7u) / 8u,
            1u);
        visibilityBarrier.before = ResourceState::ShaderRead;
        visibilityBarrier.after = ResourceState::ColorAttachment;
        context.commandBuffer().barrier(BarrierDesc{
            .textures = &visibilityBarrier,
            .textureCount = 1,
        });
        return {};
    }

    Result drawComposite(RenderGraphExecutionContext& context, TextureHandle color)
    {
        BufferBarrierDesc colorBarrier{
            .buffer = deferredColorBuffer_.get(),
            .before = deferredColorState_,
            .after = ResourceState::ShaderRead,
        };
        context.commandBuffer().barrier(BarrierDesc{
            .buffers = &colorBarrier,
            .bufferCount = 1,
        });
        deferredColorState_ = ResourceState::ShaderRead;
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
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(frameWidth_),
            .height = static_cast<float>(frameHeight_),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        context.commandBuffer().bindBindlessHeap(*streamRuntime_.bindlessHeap());
        context.commandBuffer().bindGraphicsPipeline(*compositePipeline_);
        const MeshletStreamUserPush push = streamRuntime_.userPush();
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
        context.commandBuffer().draw(3u, 1u, 0u, 0u);
        context.commandBuffer().endRendering();
        return {};
    }

    Result drawRayQuery(
        RenderGraphExecutionContext& context,
        TextureHandle color,
        const MeshletStreamFrameDesc& frame)
    {
        if (!streamRuntime_.tlasReady() || streamRuntime_.tlasHandle() == 0 || color.view() == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        SceneRayQueryVisualizationPush push;
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
        push.upProjection[3] = 0.0f;
        push.viewport[0] = static_cast<float>(std::max(context.width(), 1u)) /
            static_cast<float>(std::max(context.height(), 1u));
        push.viewport[1] = static_cast<float>(context.width());
        push.viewport[2] = static_cast<float>(context.height());
        push.viewport[3] = frame.camera.fovDegrees * 0.017453292519943295f;
        push.clipOrtho[0] = frame.camera.znear;
        push.clipOrtho[1] = frame.camera.zfar;
        push.clipOrtho[2] = std::max(streamRuntime_.bounds().radius() * 2.0f, 0.001f);
        push.clipOrtho[3] = 0.0f;
        push.mode = rtasGranularityFromProperties(context.properties());
        push.width = context.width();
        push.height = context.height();

        const vulkan::SceneRayQueryDispatchBinding bindings[] = {
            vulkan::SceneRayQueryDispatchBinding{
                .binding = 0,
                .accelerationStructureHandle = streamRuntime_.tlasHandle(),
            },
            vulkan::SceneRayQueryDispatchBinding{
                .binding = 1,
                .textureView = color.view(),
            },
        };
        return rayQueryProgram_.dispatch(vulkan::SceneRayQueryDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7u) / 8u,
            .groupCountY = (context.height() + 7u) / 8u,
            .groupCountZ = 1,
        });
    }

    MeshletStreamRuntime streamRuntime_;
    std::unique_ptr<ShaderModule> meshShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<ShaderModule> deferredShader_;
    std::unique_ptr<ShaderModule> compositeVertexShader_;
    std::unique_ptr<ShaderModule> compositeFragmentShader_;
    std::unique_ptr<ShaderModule> cullResetShader_;
    std::unique_ptr<ShaderModule> instanceCullShader_;
    std::unique_ptr<ShaderModule> hzbShader_;
    std::unique_ptr<GraphicsPipeline> visibilityPipeline_;
    std::unique_ptr<ComputePipeline> deferredPipeline_;
    std::unique_ptr<GraphicsPipeline> compositePipeline_;
    std::unique_ptr<ComputePipeline> cullResetPipeline_;
    std::unique_ptr<ComputePipeline> instanceCullPipeline_;
    std::unique_ptr<ComputePipeline> hzbPipeline_;
    std::unique_ptr<Buffer> deferredColorBuffer_;
    BindlessHandle visibilityImageHandle_;
    BindlessHandle depthImageHandle_;
    BindlessHandle deferredColorHandle_;
    BindlessHandle instanceVisibilityHandle_;
    BindlessHandle visibleInstanceIdsHandle_;
    BindlessHandle visibleInstanceCounterHandle_;
    std::array<BindlessHandle, 2> hzbHandles_{};
    ResourceState deferredColorState_ = ResourceState::Undefined;
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
    const scene::Scene* gpuSceneSource_ = nullptr;
    GPUSceneSourceOverrideToken gpuSceneSourceToken_;
    bool hzbValid_ = false;
    vulkan::SceneRayQueryProgram rayQueryProgram_;
    bool rtasVisualization_ = false;
};

std::unique_ptr<RenderGraphPass> createGPUDrivenStreamAssetPass()
{
    return std::make_unique<GPUDrivenStreamAssetPass>();
}

} // namespace metallic::render::builtin_pass
