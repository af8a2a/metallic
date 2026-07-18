#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSceneRtx.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetConfig.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

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
        return kMeshletStreamDebugPage;
    }
    const std::string mode = iter->get<std::string>();
    if (mode == "lod") {
        return kMeshletStreamDebugLod;
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
    return kMeshletStreamDebugPage;
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

} // namespace

class GPUDrivenStreamAssetPass final : public UnsafePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        RenderGraphField& color = reflection.addTextureOutput(
            "color",
            "Meshlet streamasset debug color");
        if (boolProperty(properties(), "rtasVisualization", false)) {
            color.texture2D().storageReadWrite().format = Format::Rgba8Unorm;
        } else {
            color.texture2D().colorWrite();
        }
        reflection.addTextureOutput("depth", "Meshlet streamasset debug depth")
            .texture2D()
            .depthStencilWrite();
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeBoolSetting("enableGpuLodSelection", "GPU LOD", true),
            runtimeIntSetting("selectedLodLevel", "LOD", 0, 0, 31),
            runtimeEnumSetting(
                "debugColorMode",
                "Color",
                "page",
                {
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

        ShaderCompileResult fragmentCompile;
        result = compileSlangShader(
            kMeshletStreamShaderModuleName,
            kMeshletStreamFragmentEntryPoint,
            fragmentCompile,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createShaderModule(
            ShaderModuleDesc{
                .code = fragmentCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(fragmentCompile.spirv.size() * sizeof(uint32_t)),
            },
            fragmentShader_);
        if (!result || fragmentShader_ == nullptr) {
            log += resultMessage("createShaderModule(GPUDrivenStreamAsset fragment)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .meshShader = meshShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = context.defaultFormat,
                .depthStencilFormat = Format::D32Sfloat,
                .depthStencil = DepthStencilState{
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = depthCompareOp(kDefaultReversedZ),
                },
                .usesBindlessHeap = true,
            },
            pipeline_);
        if (!result || pipeline_ == nullptr) {
            log += resultMessage("createGraphicsPipeline(GPUDrivenStreamAssetPass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        if (rtasVisualization_) {
            result = initializeRayQuery(*context.device, log);
            if (!result) {
                return result;
            }
        }

        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
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
        TextureHandle color = context.outputTexture("color");
        TextureHandle depth = context.outputTexture("depth");
        if (!color.valid() ||
            !depth.valid() ||
            context.streamer() == nullptr ||
            !streamRuntime_.ready() ||
            streamRuntime_.bindlessHeap() == nullptr ||
            pipeline_ == nullptr ||
            (rtasVisualization_ && !rayQueryProgram_.valid())) {
            return makeError(Error::InvalidArgument);
        }

        const MeshletStreamFrameDesc frame = frameDescFromContext(context);
        Result result = streamRuntime_.cmdBeginFrame(context.commandBuffer(), *context.streamer(), frame);
        if (!result) {
            return result;
        }
        result = streamRuntime_.cmdPreTraversal(context.commandBuffer(), frame);
        if (!result) {
            return result;
        }
        result = rtasVisualization_
            ? drawRayQuery(context, color, frame)
            : draw(context, color, depth);
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

    Result draw(RenderGraphExecutionContext& context, TextureHandle color, TextureHandle depth)
    {
        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.015f, 0.018f, 0.024f, 1.0f},
        };
        RenderingAttachmentDesc depthAttachment{
            .view = depth.view(),
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = LoadOp::Clear,
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
            context.commandBuffer().bindGraphicsPipeline(*pipeline_);
            const MeshletStreamUserPush push = streamRuntime_.userPush();
            context.commandBuffer().pushBindlessData(&push, sizeof(push));
            streamRuntime_.cmdDrawMeshTasks(context.commandBuffer());
        }
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
    std::unique_ptr<GraphicsPipeline> pipeline_;
    vulkan::SceneRayQueryProgram rayQueryProgram_;
    bool rtasVisualization_ = false;
};

std::unique_ptr<RenderGraphPass> createGPUDrivenStreamAssetPass()
{
    return std::make_unique<GPUDrivenStreamAssetPass>();
}

} // namespace metallic::render::builtin_pass
